"""Dense NCC verification of pHash-nominated still hits.

pHash is kept as a recall engine: it nominates a handful of timestamps per
still. Acceptance is decided by normalized cross-correlation on a short
re-decoded window, with an aspect-aware crop search so 16:9 provider stills can
line up with 2:1 letterboxed Blu-ray picture.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .cancellation import CancellationToken
from .frames import FrameExtractionError, FrameIndex, FrameIndexer, FrameStream
from .models import Episode, EpisodeScore, FileScores, MatchResult, MatchStatus, StillHit, VideoFile
from .normalize import ImageHashes, crop_borders
from .score_visual import ScoringConfig, search_candidates

__all__ = [
    "apply_duration_corroboration",
    "apply_still_uniqueness",
    "aspect_aligned_pairs",
    "best_ncc",
    "load_still_images",
    "match_by_unique_duration",
    "ncc",
    "verify_file_scores",
    "verify_scores",
]

logger = logging.getLogger(__name__)

#: Width of grayscale frames decoded for verification.
_VERIFY_WIDTH = 640
_VERIFY_HEIGHT = 360


def ncc(left: np.ndarray, right: np.ndarray, *, size: int = 64) -> float:
    """Return the normalized cross-correlation of two grayscale images.

    Both sides are border-cropped, resized to ``size`` by ``size``, mean-centred,
    and compared as unit vectors. The result is in ``[-1, 1]``; genuine
    still/frame pairs on this project's material land around 0.75, while pHash
    collisions typically sit below 0.5.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(64, dtype=np.uint8).reshape(8, 8)
    >>> round(ncc(a, a), 5)
    1.0
    >>> round(ncc(a, 255 - a), 2)
    -1.0
    """
    left_vec = _patch_vector(left, size)
    right_vec = _patch_vector(right, size)
    if left_vec is None or right_vec is None:
        return -1.0
    return float(left_vec @ right_vec)


def _patch_vector(array: np.ndarray, size: int) -> np.ndarray | None:
    """Return a zero-mean unit-norm vector, or ``None`` for a blank patch."""
    if array.size == 0:
        return None
    if array.ndim == 3:
        array = np.asarray(Image.fromarray(array).convert("L"))
    cropped = crop_borders(array)
    if cropped.size == 0:
        return None
    patch = np.asarray(
        Image.fromarray(cropped).resize((size, size), Image.Resampling.LANCZOS),
        dtype=np.float32,
    ).ravel()
    patch -= float(patch.mean())
    norm = float(np.linalg.norm(patch))
    if norm < 1e-6:
        return None
    return patch / norm


def aspect_aligned_pairs(
    still: np.ndarray, frame: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield still/frame crops that share an aspect ratio.

    Provider stills are typically 16:9 while Blu-ray picture after letterbox
    removal is often closer to 2:1. Isotropic zooms cannot close that gap;
    cropping the taller image (or the wider one) to the other aspect can.
    """
    pairs: list[tuple[np.ndarray, np.ndarray]] = [(still, frame)]
    still_h, still_w = still.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    if still_h == 0 or still_w == 0 or frame_h == 0 or frame_w == 0:
        return pairs

    still_ar = still_w / still_h
    frame_ar = frame_w / frame_h

    # Crop the still's height so its aspect matches the frame (still -> 2:1).
    target_h = round(still_w / frame_ar)
    if 8 <= target_h < still_h:
        for fraction in (0.0, 0.5, 1.0):
            y0 = round((still_h - target_h) * fraction)
            y0 = max(0, min(y0, still_h - target_h))
            pairs.append((still[y0 : y0 + target_h, :], frame))

    # Crop the frame's width so its aspect matches the still (frame -> 16:9).
    target_w = round(frame_h * still_ar)
    if 8 <= target_w < frame_w:
        for fraction in (0.0, 0.5, 1.0):
            x0 = round((frame_w - target_w) * fraction)
            x0 = max(0, min(x0, frame_w - target_w))
            pairs.append((still, frame[:, x0 : x0 + target_w]))

    # Mild center zooms on the still, for promotional overscan.
    for scale in (0.95, 0.90):
        crop_w = max(8, round(still_w * scale))
        crop_h = max(8, round(still_h * scale))
        x0 = (still_w - crop_w) // 2
        y0 = (still_h - crop_h) // 2
        pairs.append((still[y0 : y0 + crop_h, x0 : x0 + crop_w], frame))

    return pairs


def best_ncc(still: np.ndarray, frame: np.ndarray, *, size: int = 64) -> float:
    """Return the best NCC across aspect-aligned crops of ``still`` and ``frame``."""
    return max(
        (ncc(left, right, size=size) for left, right in aspect_aligned_pairs(still, frame)),
        default=-1.0,
    )


def verify_scores(
    scored: Sequence[FileScores],
    indexes: dict[Path, FrameIndex],
    episodes: Sequence[Episode],
    still_hashes: dict[str, ImageHashes],
    still_images: dict[str, np.ndarray],
    indexer: FrameIndexer,
    config: ScoringConfig,
    *,
    token: CancellationToken | None = None,
) -> list[FileScores]:
    """Re-score every file-episode pair with localized NCC verification."""
    token = token or CancellationToken()
    return [
        verify_file_scores(
            file_scores,
            indexes[file_scores.video.path],
            episodes,
            still_hashes,
            still_images,
            indexer,
            config,
            token=token,
        )
        for file_scores in scored
        if file_scores.video.path in indexes
    ]


def verify_file_scores(
    file_scores: FileScores,
    index: FrameIndex,
    episodes: Sequence[Episode],
    still_hashes: dict[str, ImageHashes],
    still_images: dict[str, np.ndarray],
    indexer: FrameIndexer,
    config: ScoringConfig,
    *,
    token: CancellationToken | None = None,
) -> FileScores:
    """Verify one file's episode scores and rewrite costs as ``1 - ncc``."""
    token = token or CancellationToken()
    video = file_scores.video
    # Reuse decoded windows when nominees cluster — common with many stills.
    window_cache: dict[float, list[tuple[float, np.ndarray]]] = {}
    verified: list[EpisodeScore] = []

    for episode in episodes:
        score = _verify_episode(
            video,
            index,
            episode,
            still_hashes,
            still_images,
            indexer,
            config,
            window_cache=window_cache,
            token=token,
        )
        verified.append(score)
        if score.ncc is not None and score.best_hit is not None:
            logger.debug(
                "%s verified %s ncc=%.3f at %.1fs (phash %d)",
                video.name,
                episode.code,
                score.ncc,
                score.best_hit.timestamp,
                score.best_hit.distance,
            )

    return FileScores(video=video, scores=tuple(verified))


def _verify_episode(
    video: VideoFile,
    index: FrameIndex,
    episode: Episode,
    still_hashes: dict[str, ImageHashes],
    still_images: dict[str, np.ndarray],
    indexer: FrameIndexer,
    config: ScoringConfig,
    *,
    window_cache: dict[float, list[tuple[float, np.ndarray]]],
    token: CancellationToken,
) -> EpisodeScore:
    """Nominate pHash peaks for ``episode`` and keep the strongest NCC."""
    best_ncc_value = -1.0
    best_hit: StillHit | None = None
    supporting_urls: set[str] = set()
    support_floor = min(config.verify_threshold, config.verify_soft_threshold)

    for still in episode.stills:
        hashes = still_hashes.get(still.url)
        still_image = still_images.get(still.url)
        if hashes is None or still_image is None:
            continue

        for phash_distance, dhash_distance, timestamp in search_candidates(
            index,
            hashes,
            k=config.verify_candidates,
            min_separation_s=config.verify_separation_s,
        ):
            token.raise_if_cancelled()
            peak = _ncc_at_timestamp(
                video,
                still_image,
                timestamp,
                indexer,
                config,
                window_cache=window_cache,
                token=token,
            )
            if peak is None:
                continue
            ncc_value, verified_timestamp = peak
            if ncc_value >= support_floor:
                supporting_urls.add(still.url)
            if ncc_value > best_ncc_value:
                best_ncc_value = ncc_value
                best_hit = StillHit(
                    still=still,
                    distance=phash_distance,
                    timestamp=verified_timestamp,
                    dhash_distance=dhash_distance,
                )

    if best_hit is None or best_ncc_value < 0:
        return EpisodeScore(
            episode=episode,
            best_hit=None,
            supporting_stills=0,
            cost=float("inf"),
            ncc=None,
        )

    return EpisodeScore(
        episode=episode,
        best_hit=best_hit,
        supporting_stills=len(supporting_urls),
        cost=max(0.0, 1.0 - best_ncc_value),
        ncc=best_ncc_value,
    )


def _ncc_at_timestamp(
    video: VideoFile,
    still_image: np.ndarray,
    timestamp: float,
    indexer: FrameIndexer,
    config: ScoringConfig,
    *,
    window_cache: dict[float, list[tuple[float, np.ndarray]]],
    token: CancellationToken,
) -> tuple[float, float] | None:
    """Decode a short window around ``timestamp`` and return ``(ncc, time)``."""
    frames = _cached_verify_window(
        video,
        timestamp,
        indexer,
        config,
        window_cache=window_cache,
        token=token,
    )
    if not frames:
        return None

    best = -1.0
    best_time = timestamp
    for frame_time, frame in frames:
        value = best_ncc(still_image, frame, size=config.verify_patch_size)
        if value > best:
            best = value
            best_time = frame_time
    return best, best_time


def _cached_verify_window(
    video: VideoFile,
    timestamp: float,
    indexer: FrameIndexer,
    config: ScoringConfig,
    *,
    window_cache: dict[float, list[tuple[float, np.ndarray]]],
    token: CancellationToken,
) -> list[tuple[float, np.ndarray]]:
    """Return frames around ``timestamp``, reusing nearby decodes when possible."""
    reuse_radius = config.verify_window_s
    for center, frames in window_cache.items():
        if abs(center - timestamp) <= reuse_radius:
            lo = timestamp - config.verify_window_s
            hi = timestamp + config.verify_window_s
            clipped = [(time, frame) for time, frame in frames if lo <= time <= hi]
            if clipped:
                return clipped

    start = max(0.0, timestamp - config.verify_window_s)
    end = timestamp + config.verify_window_s
    try:
        frames = list(
            _iter_verify_frames(
                video.path,
                start_s=start,
                duration_s=max(config.verify_interval_s, end - start),
                interval_s=config.verify_interval_s,
                hwaccel=indexer.hwaccel,
                token=token,
            )
        )
    except (FrameExtractionError, OSError) as error:
        logger.debug("verify decode failed for %s at %.1fs (%s)", video.name, timestamp, error)
        return []

    window_cache[timestamp] = frames
    return frames


def _iter_verify_frames(
    path: Path,
    *,
    start_s: float,
    duration_s: float,
    interval_s: float,
    hwaccel: str | None,
    token: CancellationToken,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(absolute_timestamp, grayscale_frame)`` for a short window."""
    stream = FrameStream(
        path,
        start_s=start_s,
        duration_s=duration_s,
        interval_s=interval_s,
        size=(_VERIFY_WIDTH, _VERIFY_HEIGHT),
        hwaccel=hwaccel,
        keyframes_only=False,
        token=token,
    )
    frames = list(stream)
    # FrameStream fills timestamps after iteration completes.
    times = stream.timestamps
    if len(times) != len(frames):
        times = [start_s + index * interval_s for index in range(len(frames))]
    yield from zip(times, frames, strict=False)


def load_still_images(still_paths: dict[str, Path]) -> dict[str, np.ndarray]:
    """Load still JPEGs as grayscale arrays keyed by URL."""
    images: dict[str, np.ndarray] = {}
    for url, path in still_paths.items():
        try:
            with Image.open(path) as image:
                images[url] = np.asarray(image.convert("L"))
        except (UnidentifiedImageError, OSError) as error:
            logger.warning("could not load still %s (%s)", path, error)
    return images


def apply_still_uniqueness(
    scored: Sequence[FileScores],
    *,
    floor: float = 0.5,
    lock_threshold: float = 0.65,
) -> list[FileScores]:
    """Drop weaker files that share a still only when the winner is decisive.

    Soft multi-file peaks (common on sparse stills) must not erase each other's
    evidence: a 0.55 claim on the wrong file previously starved the right file
    of the same still. Only a claim at or above ``lock_threshold`` locks the
    still for its file; below that, every claim is left for the Hungarian step.
    """
    # still_url -> list of (file_index, episode_number, ncc)
    claims: dict[str, list[tuple[int, int, float]]] = {}
    for file_index, file_scores in enumerate(scored):
        for score in file_scores.scores:
            if score.best_hit is None or score.ncc is None or score.ncc < floor:
                continue
            claims.setdefault(score.best_hit.still.url, []).append(
                (file_index, score.episode.number, score.ncc)
            )

    # url -> (file_index, episode_number) of a decisive winning claim.
    winners: dict[str, tuple[int, int]] = {}
    for url, entries in claims.items():
        if len(entries) <= 1:
            continue
        best_file, best_episode, best_ncc = max(entries, key=lambda entry: entry[2])
        if best_ncc < lock_threshold:
            continue
        winners[url] = (best_file, best_episode)

    if not winners:
        return list(scored)

    updated: list[FileScores] = []
    for file_index, file_scores in enumerate(scored):
        new_scores = []
        for score in file_scores.scores:
            if score.best_hit is None or score.ncc is None:
                new_scores.append(score)
                continue
            url = score.best_hit.still.url
            winner = winners.get(url)
            if winner is None or winner == (file_index, score.episode.number):
                new_scores.append(score)
                continue
            logger.debug(
                "dropped shared still on %s/%s: ncc %.3f (winner locked at >= %.2f)",
                file_scores.video.name,
                score.episode.code,
                score.ncc,
                lock_threshold,
            )
            new_scores.append(
                replace(
                    score,
                    best_hit=None,
                    supporting_stills=0,
                    cost=float("inf"),
                    ncc=None,
                )
            )
        updated.append(FileScores(video=file_scores.video, scores=tuple(new_scores)))
    return updated


def apply_duration_corroboration(
    scored: Sequence[FileScores],
    *,
    match_tolerance_s: float = 90.0,
    mismatch_s: float = 120.0,
    boost: float = 0.12,
    penalty: float = 0.15,
) -> list[FileScores]:
    """Adjust NCC using episode runtime when it agrees or conflicts with the file.

    Provider stills on later discs are often too sparse for NCC alone. Runtime is
    never used to invent a hit, only to nudge an existing verification: matching
    durations get a small boost, clear mismatches a penalty. When two episodes
    share a runtime the adjustments cancel and NCC remains decisive.
    """
    updated: list[FileScores] = []
    for file_scores in scored:
        duration = file_scores.video.duration_s
        new_scores = []
        for score in file_scores.scores:
            runtime = score.episode.runtime_minutes
            if score.ncc is None or runtime is None or runtime <= 0:
                new_scores.append(score)
                continue
            delta = abs(duration - runtime * 60.0)
            ncc_value = score.ncc
            if delta <= match_tolerance_s:
                ncc_value = min(1.0, ncc_value + boost)
            elif delta >= mismatch_s:
                ncc_value = ncc_value - penalty
            new_scores.append(
                replace(
                    score,
                    ncc=ncc_value,
                    cost=max(0.0, 1.0 - ncc_value) if ncc_value >= 0 else float("inf"),
                )
            )
        updated.append(FileScores(video=file_scores.video, scores=tuple(new_scores)))
    return updated


def match_by_unique_duration(
    results: Sequence[MatchResult],
    episodes: Sequence[Episode],
    *,
    tolerance_s: float = 90.0,
) -> list[MatchResult]:
    """Fill unmatched closed-world slots by closest unique episode runtime.

    Each leftover file picks its closest leftover episode whose runtime is
    within ``tolerance_s``. The picks are applied only when that mapping is
    injective (no two files claim the same episode). Partial fills are allowed.
    """
    taken = {
        result.episode.number
        for result in results
        if result.status is MatchStatus.MATCHED and result.episode is not None
    }
    leftover_episodes = [episode for episode in episodes if episode.number not in taken]
    leftover_results = [
        (index, result)
        for index, result in enumerate(results)
        if result.status is MatchStatus.UNMATCHED
    ]
    if not leftover_results or not leftover_episodes:
        return list(results)

    picks: dict[int, Episode] = {}
    for index, result in leftover_results:
        duration = result.video.duration_s
        ranked = sorted(
            (
                episode
                for episode in leftover_episodes
                if episode.runtime_minutes and episode.runtime_minutes > 0
            ),
            key=lambda episode: abs(duration - episode.runtime_minutes * 60.0),
        )
        if not ranked:
            continue
        best = ranked[0]
        if abs(duration - best.runtime_minutes * 60.0) > tolerance_s:
            continue
        picks[index] = best

    claimed_numbers = [episode.number for episode in picks.values()]
    if not picks or len(claimed_numbers) != len(set(claimed_numbers)):
        return list(results)

    updated = list(results)
    for index, episode in picks.items():
        previous = updated[index]
        updated[index] = MatchResult(
            video=previous.video,
            status=MatchStatus.MATCHED,
            episode=episode,
            hit=previous.hit,
            cost=previous.cost,
            runner_up=previous.runner_up,
            runner_up_cost=previous.runner_up_cost,
            supporting_stills=previous.supporting_stills,
            confidence=0.5,
            ncc=previous.ncc,
            notes=[
                f"matched by unique runtime ({episode.runtime_minutes} min); "
                "visual stills were inconclusive"
            ],
        )
        logger.info(
            "duration-fallback %s -> %s (%.0fs ≈ %s min)",
            previous.video.name,
            episode.code,
            previous.video.duration_s,
            episode.runtime_minutes,
        )
    return updated
