"""Search every provider still against every file's frame index.

This is the primary matcher. Duration never enters here: an episode is
identified because one of its published frames was found inside a particular
rip, at a particular timestamp a human can jump to and verify.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError

from .cache import ImageCache, JsonCache
from .cancellation import CancellationToken
from .frames import FrameIndex
from .models import Episode, EpisodeScore, FileScores, Still, StillHit, VideoFile
from .normalize import ImageHashes, hamming_distances, hash_image

__all__ = [
    "ScoringConfig",
    "episode_cost",
    "hash_stills",
    "score_all",
    "score_episode",
    "score_file",
    "search_candidates",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Thresholds governing when a still counts as found inside a file.

    Attributes
    ----------
    match_threshold
        Maximum pHash Hamming distance (``T``) for a still to count as a hit.
        Around 10-12 of 64 bits tolerates re-encoding and rescaling while still
        rejecting unrelated shots. Used for nomination support counts, and as
        the accept threshold when verification is off.
    decision_gap
        How much worse (``G``) the runner-up episode must be before a match is
        called confident rather than ambiguous, in pHash cost units.
    agreement_bonus, max_agreement_bonus
        Each additional agreeing still shaves ``agreement_bonus`` off the cost,
        up to ``max_agreement_bonus``. Several stills of the same episode
        landing in the same file is much stronger evidence than one.
    dhash_threshold, dhash_penalty
        A hit whose dHash disagrees this badly is treated as a probable pHash
        collision: it stops counting as support and its cost is penalised.
    verify
        When true, pHash only nominates candidate timestamps and a dense NCC
        pass decides acceptance.
    verify_candidates
        How many temporally separated pHash peaks to reconsider per still.
    verify_separation_s
        Minimum seconds between nominated peaks for the same still.
    verify_window_s
        Half-width of the dense re-decode window around each nominee.
    verify_interval_s
        Sampling interval inside the verification window.
    verify_threshold
        Minimum NCC for a pair to be feasible when verification is on.
    verify_gap
        Mutual-best margin in NCC units (``cost = 1 - ncc``).
    verify_patch_size
        Side length of the square patch used for NCC.

    Examples
    --------
    >>> ScoringConfig().match_threshold
    12
    """

    match_threshold: int = 12
    decision_gap: float = 4.0
    agreement_bonus: float = 0.75
    max_agreement_bonus: float = 3.0
    dhash_threshold: int = 22
    dhash_penalty: float = 6.0
    verify: bool = True
    verify_candidates: int = 8
    verify_separation_s: float = 30.0
    verify_window_s: float = 0.6
    verify_interval_s: float = 0.1
    verify_threshold: float = 0.65
    verify_gap: float = 0.05
    verify_patch_size: int = 64

    def __post_init__(self) -> None:
        """Validate the thresholds."""
        if not 0 <= self.match_threshold <= 64:
            raise ValueError("match_threshold must be between 0 and 64 bits")
        if self.decision_gap < 0:
            raise ValueError("decision_gap must not be negative")
        if self.verify_candidates < 1:
            raise ValueError("verify_candidates must be at least 1")
        if self.verify_separation_s < 0:
            raise ValueError("verify_separation_s must not be negative")
        if self.verify_window_s <= 0:
            raise ValueError("verify_window_s must be positive")
        if self.verify_interval_s <= 0:
            raise ValueError("verify_interval_s must be positive")
        if not 0.0 <= self.verify_threshold <= 1.0:
            raise ValueError("verify_threshold must be between 0 and 1")
        if self.verify_gap < 0:
            raise ValueError("verify_gap must not be negative")
        if self.verify_patch_size < 8:
            raise ValueError("verify_patch_size must be at least 8")


def hash_stills(
    stills: Iterable[Still],
    *,
    client: httpx.Client,
    images: ImageCache,
    hash_cache: JsonCache | None = None,
    max_workers: int = 8,
    token: CancellationToken | None = None,
) -> dict[str, ImageHashes]:
    """Download and hash provider stills, keyed by URL.

    Stills that cannot be downloaded or decoded are omitted: less evidence is
    a normal outcome, not an error.

    Parameters
    ----------
    stills
        Stills to prepare. Duplicate URLs are handled once.
    client
        HTTP client used for downloads.
    images
        Disk cache for the image bytes.
    hash_cache
        Optional cache of the computed hashes, so a repeat run skips both the
        download and the decode.
    max_workers
        Concurrency for downloading.

    Returns
    -------
    dict
        Maps still URL to its :class:`~mkv_episode_matcher.normalize.ImageHashes`.
    """
    unique: dict[str, Still] = {still.url: still for still in stills}
    if not unique:
        return {}
    token = token or CancellationToken()
    token.raise_if_cancelled()

    hashed: dict[str, ImageHashes] = {}
    pending: dict[str, Still] = {}
    for url, still in unique.items():
        cached = _cached_hashes(hash_cache, url)
        if cached is not None:
            hashed[url] = cached
        else:
            pending[url] = still

    if pending:

        def prepare(still: Still) -> ImageHashes | None:
            if token.cancelled:
                return None
            return _download_and_hash(client, images, still)

        pool = ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(pending))))
        try:
            futures = {pool.submit(prepare, still): url for url, still in pending.items()}
            for future, url in futures.items():
                result = future.result()
                if result is None:
                    continue
                hashed[url] = result
                if hash_cache is not None:
                    hash_cache.put(
                        f"still-hash/{url}", {"phash": result.phash, "dhash": result.dhash}
                    )
        finally:
            pool.shutdown(wait=True, cancel_futures=token.cancelled)
        token.raise_if_cancelled()

    missing = len(unique) - len(hashed)
    if missing:
        logger.warning("%d of %d stills could not be hashed", missing, len(unique))
    logger.info("prepared %d still hashes", len(hashed))
    return hashed


def _cached_hashes(hash_cache: JsonCache | None, url: str) -> ImageHashes | None:
    """Return previously computed hashes for ``url``, if any."""
    if hash_cache is None:
        return None
    payload = hash_cache.get(f"still-hash/{url}")
    if not payload:
        return None
    try:
        return ImageHashes(phash=int(payload["phash"]), dhash=int(payload["dhash"]))
    except (KeyError, TypeError, ValueError):
        return None


def _download_and_hash(
    client: httpx.Client, images: ImageCache, still: Still
) -> ImageHashes | None:
    """Fetch one still and hash it, or return ``None`` if that is not possible."""
    path = images.fetch(client, still)
    if path is None:
        return None
    try:
        with Image.open(path) as image:
            return hash_image(image)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        logger.warning("could not decode still %s (%s)", still.url, error)
        return None


def episode_cost(
    best_distance: int | None, *, supporting_stills: int, config: ScoringConfig
) -> float:
    """Turn a best distance and an agreement count into an assignment cost.

    Lower is better. The bonus for corroborating stills is deliberately small
    relative to ``match_threshold``: agreement breaks ties, it does not rescue
    a weak hit.

    Examples
    --------
    >>> episode_cost(6, supporting_stills=1, config=ScoringConfig())
    6.0
    >>> episode_cost(None, supporting_stills=0, config=ScoringConfig())
    inf
    """
    if best_distance is None:
        return float("inf")
    bonus = min(config.agreement_bonus * max(0, supporting_stills - 1), config.max_agreement_bonus)
    return max(0.0, float(best_distance) - bonus)


def score_episode(
    index: FrameIndex,
    episode: Episode,
    still_hashes: dict[str, ImageHashes],
    config: ScoringConfig,
) -> EpisodeScore:
    """Search every still of ``episode`` against one file's frame index.

    Parameters
    ----------
    index
        The file's dense frame-hash index.
    episode
        The episode whose stills are being searched for.
    still_hashes
        URL to hashes, as returned by :func:`hash_stills`.
    config
        Thresholds.

    Returns
    -------
    EpisodeScore
        The strongest hit, how many stills agreed, and the assignment cost.
    """
    best: StillHit | None = None
    supporting = 0

    for still in episode.stills:
        hashes = still_hashes.get(still.url)
        if hashes is None:
            continue
        found = _search(index, hashes)
        if found is None:
            continue
        phash_distance, dhash_distance, timestamp = found
        if phash_distance <= config.match_threshold and dhash_distance <= config.dhash_threshold:
            supporting += 1
        if best is None or phash_distance < best.distance:
            best = StillHit(
                still=still,
                distance=phash_distance,
                timestamp=timestamp,
                dhash_distance=dhash_distance,
            )

    cost = episode_cost(
        None if best is None else best.distance, supporting_stills=supporting, config=config
    )
    if best is not None and best.dhash_distance > config.dhash_threshold:
        cost += config.dhash_penalty
    return EpisodeScore(episode=episode, best_hit=best, supporting_stills=supporting, cost=cost)


def _search(index: FrameIndex, hashes: ImageHashes) -> tuple[int, int, float] | None:
    """Return ``(phash distance, dhash distance, timestamp)`` of the closest frame."""
    candidates = search_candidates(index, hashes, k=1, min_separation_s=0.0)
    return candidates[0] if candidates else None


def search_candidates(
    index: FrameIndex,
    hashes: ImageHashes,
    *,
    k: int = 8,
    min_separation_s: float = 30.0,
) -> list[tuple[int, int, float]]:
    """Return the strongest pHash peaks for ``hashes`` inside ``index``.

    Candidates are not gated by ``match_threshold``: a true still can sit at
    Hamming distance 14-16 after aspect mismatch and still verify under NCC.
    Temporal non-maximum suppression keeps peaks at least ``min_separation_s``
    apart so the verifier does not re-decode the same shot.

    Returns
    -------
    list of (phash_distance, dhash_distance, timestamp)
        Best first, length at most ``k``.
    """
    if len(index) == 0 or k < 1:
        return []

    distances = hamming_distances(hashes.phash, index.phashes)
    order = np.argsort(distances)
    picked: list[tuple[int, int, float]] = []
    picked_times: list[float] = []

    for position in order:
        timestamp = float(index.timestamps[position])
        if any(abs(timestamp - prior) < min_separation_s for prior in picked_times):
            continue
        dhash_distance = int(
            hamming_distances(hashes.dhash, index.dhashes[position : position + 1])[0]
        )
        picked.append((int(distances[position]), dhash_distance, timestamp))
        picked_times.append(timestamp)
        if len(picked) >= k:
            break
    return picked


def score_file(
    video: VideoFile,
    index: FrameIndex,
    episodes: Sequence[Episode],
    still_hashes: dict[str, ImageHashes],
    config: ScoringConfig,
) -> FileScores:
    """Score one file against every episode of the season."""
    scores = tuple(score_episode(index, episode, still_hashes, config) for episode in episodes)
    best = min(scores, key=lambda score: score.cost, default=None)
    if best is not None and best.best_hit is not None:
        logger.debug(
            "%s best candidate %s at %.1fs (distance %d, %d supporting stills)",
            video.name,
            best.episode.code,
            best.best_hit.timestamp,
            best.best_hit.distance,
            best.supporting_stills,
        )
    return FileScores(video=video, scores=scores)


def score_all(
    videos: Sequence[VideoFile],
    indexes: dict[Path, FrameIndex],
    episodes: Sequence[Episode],
    still_hashes: dict[str, ImageHashes],
    config: ScoringConfig,
) -> list[FileScores]:
    """Score every indexed file against the season.

    Files without an index (because decoding failed) are left out; the caller
    reports them separately rather than pretending they were searched.
    """
    results = []
    for video in videos:
        index = indexes.get(video.path)
        if index is None:
            logger.warning("no frame index for %s; leaving it unmatched", video.name)
            continue
        results.append(score_file(video, index, episodes, still_hashes, config))
    return results
