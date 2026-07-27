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
from .models import Episode, EpisodeScore, FileScores, StillHit, VideoFile
from .normalize import ImageHashes, crop_borders
from .score_visual import ScoringConfig, search_candidates

__all__ = [
    "apply_still_uniqueness",
    "aspect_aligned_pairs",
    "best_ncc",
    "load_still_images",
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
    >>> round(ncc(a, 255 - a), 5)
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
    token: CancellationToken,
) -> EpisodeScore:
    """Nominate pHash peaks for ``episode`` and keep the strongest NCC."""
    best_ncc_value = -1.0
    best_hit: StillHit | None = None
    supporting_urls: set[str] = set()

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
                token=token,
            )
            if peak is None:
                continue
            ncc_value, verified_timestamp = peak
            if ncc_value >= config.verify_threshold:
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
    token: CancellationToken,
) -> tuple[float, float] | None:
    """Decode a short window around ``timestamp`` and return ``(ncc, time)``."""
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
        return None

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


def apply_still_uniqueness(scored: Sequence[FileScores], *, floor: float = 0.5) -> list[FileScores]:
    """Demote stills that verify strongly against many files.

    A promotional or establishing shot that scores above ``floor`` on several
    rips is treated as non-discriminative: each file's NCC for that still is
    scaled by ``1 / count``. Unique stills (count 1) are unchanged.
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

    weight_by_url = {
        url: 1.0 / len(entries) if len(entries) > 1 else 1.0 for url, entries in claims.items()
    }
    if all(weight == 1.0 for weight in weight_by_url.values()):
        return list(scored)

    updated: list[FileScores] = []
    for file_scores in scored:
        new_scores = []
        for score in file_scores.scores:
            if score.best_hit is None or score.ncc is None:
                new_scores.append(score)
                continue
            weight = weight_by_url.get(score.best_hit.still.url, 1.0)
            if weight >= 1.0:
                new_scores.append(score)
                continue
            adjusted = score.ncc * weight
            new_scores.append(
                replace(
                    score,
                    ncc=adjusted,
                    cost=max(0.0, 1.0 - adjusted),
                    supporting_stills=0 if adjusted < floor else score.supporting_stills,
                )
            )
            logger.debug(
                "demoted shared still on %s/%s: ncc %.3f -> %.3f (weight %.2f)",
                file_scores.video.name,
                score.episode.code,
                score.ncc,
                adjusted,
                weight,
            )
        updated.append(FileScores(video=file_scores.video, scores=tuple(new_scores)))
    return updated
