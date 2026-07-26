"""Dense frame sampling and the persistent per-file hash index.

A provider still can sit anywhere in a 45-minute episode, which is exactly why
matching one by hand is slow. The fix is to make the whole episode searchable:
decode it once on a fixed cadence, hash every sampled frame, and keep the
result on disk. Re-matching the same disc afterwards costs a few milliseconds
of Hamming arithmetic instead of another full decode.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from .models import VideoFile
from .normalize import hamming_distances, hash_array
from .probe import probe_dimensions

__all__ = [
    "FrameExtractionError",
    "FrameIndex",
    "FrameIndexCache",
    "FrameIndexer",
    "IndexParams",
    "build_indexes",
    "extract_frame_hashes",
    "grab_frame",
]

logger = logging.getLogger(__name__)

#: Bumped whenever the on-disk index layout or the hashing pipeline changes.
INDEX_FORMAT_VERSION = 1


class FrameExtractionError(RuntimeError):
    """Raised when ``ffmpeg`` fails to produce frames for a file."""


def require_ffmpeg() -> str:
    """Return the path to ``ffmpeg``, raising if it is not installed."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise FrameExtractionError("ffmpeg was not found on PATH; install it to continue")
    return executable


@dataclass(frozen=True, slots=True)
class IndexParams:
    """Sampling settings that define -- and key -- a frame-hash index.

    Attributes
    ----------
    interval_s
        Seconds between sampled frames. One to two seconds is dense enough that
        any still lands within half an interval of a sampled frame.
    skip_head_s, skip_tail_s
        Runtime trimmed from each end before sampling. Left at zero by default:
        provider stills are drawn from cold opens and end tags too.
    sample_width
        Width frames are scaled to before hashing. Downscaling in ``ffmpeg`` is
        far cheaper than doing it in Python, and the hash only needs 32x32.

    Examples
    --------
    >>> IndexParams().interval_s
    1.0
    >>> IndexParams(interval_s=2.0).fingerprint != IndexParams().fingerprint
    True
    """

    interval_s: float = 1.0
    skip_head_s: float = 0.0
    skip_tail_s: float = 0.0
    sample_width: int = 320

    def __post_init__(self) -> None:
        """Validate the sampling settings."""
        if self.interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if self.sample_width < 32:
            raise ValueError("sample_width must be at least 32 pixels")
        if self.skip_head_s < 0 or self.skip_tail_s < 0:
            raise ValueError("skip_head_s and skip_tail_s must not be negative")

    @property
    def fingerprint(self) -> str:
        """Return a short digest identifying this parameter set."""
        payload = json.dumps(
            {
                "version": INDEX_FORMAT_VERSION,
                "interval_s": round(self.interval_s, 6),
                "skip_head_s": round(self.skip_head_s, 6),
                "skip_tail_s": round(self.skip_tail_s, 6),
                "sample_width": self.sample_width,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(slots=True)
class FrameIndex:
    """Perceptual hashes of every sampled frame of one video file.

    Attributes
    ----------
    timestamps
        ``float32`` seconds, one per sampled frame.
    phashes, dhashes
        ``uint64`` hashes aligned with ``timestamps``.
    source_size, source_mtime_ns
        Snapshot of the video's ``stat`` at index time, used to detect that the
        file changed underneath a cached index.
    """

    video_path: Path
    params: IndexParams
    timestamps: np.ndarray
    phashes: np.ndarray
    dhashes: np.ndarray
    source_size: int
    source_mtime_ns: int

    def __len__(self) -> int:
        """Return the number of sampled frames."""
        return int(self.timestamps.size)

    def is_valid_for(self, video: VideoFile, params: IndexParams) -> bool:
        """Return whether this index still describes ``video`` under ``params``."""
        return (
            self.source_size == video.size_bytes
            and self.source_mtime_ns == video.mtime_ns
            and self.params == params
        )

    def nearest(self, phash: int) -> tuple[int, int] | None:
        """Return the ``(position, distance)`` of the closest frame to ``phash``.

        Parameters
        ----------
        phash
            A 64-bit perceptual hash, typically of a provider still.

        Returns
        -------
        tuple of (int, int) or None
            Index into :attr:`timestamps` and the Hamming distance, or ``None``
            when the index is empty.
        """
        if len(self) == 0:
            return None
        distances = hamming_distances(phash, self.phashes)
        position = int(np.argmin(distances))
        return position, int(distances[position])

    def save(self, path: Path) -> None:
        """Write this index to ``path`` as a compressed ``.npz`` archive."""
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": INDEX_FORMAT_VERSION,
            "video_path": str(self.video_path),
            "source_size": self.source_size,
            "source_mtime_ns": self.source_mtime_ns,
            "params": {
                "interval_s": self.params.interval_s,
                "skip_head_s": self.params.skip_head_s,
                "skip_tail_s": self.params.skip_tail_s,
                "sample_width": self.params.sample_width,
            },
        }
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            timestamps=self.timestamps,
            phashes=self.phashes,
            dhashes=self.dhashes,
            meta=np.array(json.dumps(meta)),
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> FrameIndex:
        """Read an index previously written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"].item()))
            if meta.get("version") != INDEX_FORMAT_VERSION:
                raise ValueError(f"unsupported index version in {path}")
            return cls(
                video_path=Path(meta["video_path"]),
                params=IndexParams(**meta["params"]),
                timestamps=archive["timestamps"],
                phashes=archive["phashes"],
                dhashes=archive["dhashes"],
                source_size=int(meta["source_size"]),
                source_mtime_ns=int(meta["source_mtime_ns"]),
            )


def _scaled_size(source: tuple[int, int], sample_width: int) -> tuple[int, int]:
    """Return an even-sided target size preserving the source aspect ratio.

    Examples
    --------
    >>> _scaled_size((1920, 1080), 320)
    (320, 180)
    >>> _scaled_size((720, 576), 320)
    (320, 256)
    """
    source_width, source_height = source
    width = min(sample_width, source_width)
    width -= width % 2
    height = max(2, round(source_height * width / source_width))
    height -= height % 2
    return max(2, width), height


def iter_raw_frames(
    path: Path,
    *,
    start_s: float,
    duration_s: float | None,
    interval_s: float,
    size: tuple[int, int],
) -> Iterator[np.ndarray]:
    """Yield grayscale frames decoded from ``path`` on a fixed cadence.

    A single ``ffmpeg`` process streams raw frames over a pipe, so the file is
    decoded exactly once no matter how many frames are wanted.

    Parameters
    ----------
    path
        Video file to decode.
    start_s
        Offset to start sampling from.
    duration_s
        How much runtime to cover, or ``None`` for "to the end".
    interval_s
        Seconds between sampled frames.
    size
        ``(width, height)`` the frames are scaled to.

    Yields
    ------
    numpy.ndarray
        ``uint8`` arrays of shape ``(height, width)``.
    """
    width, height = size
    command = [require_ffmpeg(), "-hide_banner", "-nostdin", "-v", "error"]
    if start_s > 0:
        command += ["-ss", f"{start_s:.3f}"]
    command += ["-i", str(path)]
    if duration_s is not None:
        command += ["-t", f"{duration_s:.3f}"]
    command += [
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"fps=1/{interval_s:.6f},scale={width}:{height}",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    logger.debug("extracting frames: %s", " ".join(command))

    frame_bytes = width * height
    with tempfile.TemporaryFile() as error_sink:
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command,
            stdout=subprocess.PIPE,
            stderr=error_sink,
            stdin=subprocess.DEVNULL,
            bufsize=frame_bytes,
        )
        try:
            assert process.stdout is not None
            while True:
                payload = process.stdout.read(frame_bytes)
                if len(payload) < frame_bytes:
                    break
                yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width)
        finally:
            with contextlib.suppress(OSError):
                if process.stdout is not None:
                    process.stdout.close()
            returncode = process.wait()
        if returncode != 0:
            error_sink.seek(0)
            detail = error_sink.read().decode(errors="replace").strip()
            raise FrameExtractionError(f"ffmpeg failed for {path}: {detail}")


def extract_frame_hashes(
    path: Path,
    *,
    start_s: float,
    duration_s: float | None,
    interval_s: float,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode ``path`` on a cadence and hash every sampled frame.

    Returns
    -------
    tuple of numpy.ndarray
        ``(timestamps, phashes, dhashes)``.
    """
    timestamps: list[float] = []
    phashes: list[int] = []
    dhashes: list[int] = []
    for position, frame in enumerate(
        iter_raw_frames(
            path,
            start_s=start_s,
            duration_s=duration_s,
            interval_s=interval_s,
            size=size,
        )
    ):
        hashes = hash_array(frame)
        timestamps.append(start_s + position * interval_s)
        phashes.append(hashes.phash)
        dhashes.append(hashes.dhash)

    return (
        np.asarray(timestamps, dtype=np.float32),
        np.asarray(phashes, dtype=np.uint64),
        np.asarray(dhashes, dtype=np.uint64),
    )


def grab_frame(path: Path, timestamp_s: float) -> Image.Image:
    """Decode the single frame of ``path`` nearest to ``timestamp_s``.

    Used for previews and spot-checks, where one frame is wanted at full
    quality rather than a whole index.

    Parameters
    ----------
    path
        Video file to read.
    timestamp_s
        Position in seconds.

    Returns
    -------
    PIL.Image.Image
        The decoded frame.

    Raises
    ------
    FrameExtractionError
        If ``ffmpeg`` produced no frame at that position.
    """
    command = [
        require_ffmpeg(),
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{max(0.0, timestamp_s):.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2",
        "-c:v",
        "png",
        "pipe:1",
    ]
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        command, capture_output=True, stdin=subprocess.DEVNULL, check=False
    )
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode(errors="replace").strip()
        raise FrameExtractionError(f"could not grab {path} at {timestamp_s:.1f}s: {detail}")
    return Image.open(io.BytesIO(completed.stdout))


class FrameIndexCache:
    """On-disk store of frame-hash indexes, keyed by file identity and params."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, video_path: Path, params: IndexParams) -> Path:
        """Return the cache file that would hold this file's index."""
        digest = hashlib.sha256(str(Path(video_path).absolute()).encode()).hexdigest()[:16]
        return self.root / f"{video_path.stem}-{digest}-{params.fingerprint}.npz"

    def load(self, video: VideoFile, params: IndexParams) -> FrameIndex | None:
        """Return the cached index for ``video``, or ``None`` if it is unusable."""
        path = self.path_for(video.path, params)
        if not path.exists():
            return None
        try:
            index = FrameIndex.load(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            logger.warning("discarding unreadable index %s (%s)", path, error)
            return None
        if not index.is_valid_for(video, params):
            logger.info("index for %s is stale; rebuilding", video.name)
            return None
        return index

    def store(self, index: FrameIndex) -> Path:
        """Persist ``index`` and return where it was written."""
        path = self.path_for(index.video_path, index.params)
        index.save(path)
        return path


class FrameIndexer:
    """Builds frame-hash indexes, reusing cached ones whenever they are valid."""

    def __init__(self, cache: FrameIndexCache, params: IndexParams) -> None:
        self.cache = cache
        self.params = params

    def get(self, video: VideoFile, *, refresh: bool = False) -> FrameIndex:
        """Return the frame index for ``video``, building it only if needed.

        Parameters
        ----------
        video
            The probed video file.
        refresh
            Ignore any cached index and rebuild from the video.
        """
        if not refresh:
            cached = self.cache.load(video, self.params)
            if cached is not None:
                logger.debug("reusing cached index for %s (%d frames)", video.name, len(cached))
                return cached

        index = self.build(video)
        self.cache.store(index)
        return index

    def build(self, video: VideoFile) -> FrameIndex:
        """Decode and hash ``video`` from scratch."""
        size = _scaled_size(probe_dimensions(video.path), self.params.sample_width)
        span = video.duration_s - self.params.skip_head_s - self.params.skip_tail_s
        duration = span if span > 0 else None
        logger.info(
            "indexing %s (%.0fs, every %.2fs, %dx%d)",
            video.name,
            video.duration_s,
            self.params.interval_s,
            *size,
        )
        timestamps, phashes, dhashes = extract_frame_hashes(
            video.path,
            start_s=self.params.skip_head_s,
            duration_s=duration,
            interval_s=self.params.interval_s,
            size=size,
        )
        if timestamps.size == 0:
            raise FrameExtractionError(f"no frames could be decoded from {video.path}")
        logger.info("indexed %s: %d frames", video.name, timestamps.size)
        return FrameIndex(
            video_path=video.path,
            params=self.params,
            timestamps=timestamps,
            phashes=phashes,
            dhashes=dhashes,
            source_size=video.size_bytes,
            source_mtime_ns=video.mtime_ns,
        )

    def extract_window(
        self, video: VideoFile, *, start_s: float, end_s: float, interval_s: float
    ) -> FrameIndex:
        """Sample a short window of ``video`` densely, without touching the cache.

        This is the refinement half of the optional two-stage mode: index the
        whole file coarsely, then re-sample only around promising hits.
        """
        start = max(0.0, start_s)
        size = _scaled_size(probe_dimensions(video.path), self.params.sample_width)
        timestamps, phashes, dhashes = extract_frame_hashes(
            video.path,
            start_s=start,
            duration_s=max(interval_s, end_s - start),
            interval_s=interval_s,
            size=size,
        )
        return FrameIndex(
            video_path=video.path,
            params=replace(self.params, interval_s=interval_s, skip_head_s=start),
            timestamps=timestamps,
            phashes=phashes,
            dhashes=dhashes,
            source_size=video.size_bytes,
            source_mtime_ns=video.mtime_ns,
        )


def build_indexes(
    indexer: FrameIndexer,
    videos: Sequence[VideoFile],
    *,
    max_workers: int = 4,
    refresh: bool = False,
) -> dict[Path, FrameIndex]:
    """Index several files concurrently.

    Decoding happens in ``ffmpeg`` subprocesses, so worker threads spend most of
    their time waiting on a pipe rather than fighting over the GIL.

    Parameters
    ----------
    indexer
        Configured :class:`FrameIndexer`.
    videos
        Files to index.
    max_workers
        Maximum number of concurrent ``ffmpeg`` processes.
    refresh
        Rebuild every index, ignoring the cache.

    Returns
    -------
    dict
        Maps each video path to its frame index. Files that fail to decode are
        omitted and logged.
    """
    if not videos:
        return {}

    indexes: dict[Path, FrameIndex] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(videos)))) as pool:
        futures = {
            pool.submit(indexer.get, video, refresh=refresh): video for video in videos
        }
        for future, video in futures.items():
            try:
                indexes[video.path] = future.result()
            except (FrameExtractionError, OSError) as error:
                logger.error("could not index %s: %s", video.name, error)
    return indexes
