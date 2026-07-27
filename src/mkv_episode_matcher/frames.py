"""Dense frame sampling and the persistent per-file hash index.

A provider still can sit anywhere in a 45-minute episode, which is exactly why
matching one by hand is slow. The fix is to make the whole episode searchable:
decode it once on a fixed cadence, hash every sampled frame, and keep the
result on disk. Re-matching the same disc afterwards costs a few milliseconds
of Hamming arithmetic instead of another full decode.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from .cancellation import CancellationToken, OperationCancelledError
from .models import VideoFile
from .normalize import hamming_distances, hash_array
from .probe import probe_stream

__all__ = [
    "FrameExtractionError",
    "FrameIndex",
    "FrameIndexCache",
    "FrameIndexer",
    "FrameStream",
    "IndexParams",
    "build_indexes",
    "extract_frame_hashes",
    "forget_hwaccel_failures",
    "grab_frame",
    "hwaccel_has_failed",
    "hwaccel_supported",
    "parse_showinfo_timestamps",
]

logger = logging.getLogger(__name__)

#: Bumped whenever the on-disk index layout or the hashing pipeline changes.
INDEX_FORMAT_VERSION = 1

#: How much coarser than the requested interval keyframe sampling may get before
#: it is worth telling the user their stills may be falling through the gaps.
_SPARSE_KEYFRAME_RATIO = 2.0


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
    keyframes_only
        Decode only keyframes. Several times cheaper, because inter frames are
        never reconstructed, but sampling is then limited to wherever the
        encoder happened to place its keyframes.

    Examples
    --------
    >>> IndexParams().interval_s
    1.0
    >>> IndexParams(interval_s=2.0).fingerprint != IndexParams().fingerprint
    True
    >>> IndexParams(keyframes_only=True).fingerprint != IndexParams().fingerprint
    True
    """

    interval_s: float = 1.0
    skip_head_s: float = 0.0
    skip_tail_s: float = 0.0
    sample_width: int = 320
    keyframes_only: bool = False

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
                "keyframes_only": self.keyframes_only,
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
                "keyframes_only": self.params.keyframes_only,
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


_PTS_TIME = re.compile(r"pts_time:(\S+)")


@functools.lru_cache(maxsize=1)
def _passthrough_flag() -> tuple[str, str]:
    """Return the argument pair that stops ffmpeg re-timing the output stream.

    Without it, ffmpeg conforms the raw output to a constant frame rate by
    duplicating frames, which would silently corrupt the index.
    """
    try:
        banner = subprocess.run(
            [require_ffmpeg(), "-version"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return "-vsync", "0"
    found = re.search(r"ffmpeg version n?(\d+)", banner)
    major = int(found.group(1)) if found else 0
    return ("-fps_mode", "passthrough") if major >= 5 else ("-vsync", "0")


def parse_showinfo_timestamps(text: str) -> list[float]:
    r"""Extract frame presentation times, in order, from ``showinfo`` output.

    Examples
    --------
    >>> parse_showinfo_timestamps("n:0 pts_time:0 x\nn:1 pts_time:3.5 y")
    [0.0, 3.5]
    >>> parse_showinfo_timestamps("no frames here")
    []
    """
    times = []
    for match in _PTS_TIME.finditer(text):
        try:
            times.append(float(match.group(1)))
        except ValueError:
            continue
    return times


class FrameStream:
    """Streams sampled frames from one ``ffmpeg`` process, with true timestamps.

    Frames are selected with the ``select`` filter rather than ``fps``. That
    distinction matters: ``fps`` resamples onto a synthetic clock and stamps
    each output frame with a time up to half an interval earlier than the
    picture it actually carries, which would send anyone spot-checking a match
    to the wrong moment. ``select`` passes real frames through untouched, and
    ``showinfo`` reports their real presentation times.

    Attributes
    ----------
    timestamps
        Populated once iteration finishes: the true time of each yielded frame,
        in seconds from the start of the file.
    """

    def __init__(
        self,
        path: Path,
        *,
        start_s: float,
        duration_s: float | None,
        interval_s: float,
        size: tuple[int, int],
        hwaccel: str | None = None,
        keyframes_only: bool = False,
        token: CancellationToken | None = None,
    ) -> None:
        self.path = path
        self.start_s = start_s
        self.duration_s = duration_s
        self.interval_s = interval_s
        self.size = size
        self.hwaccel = hwaccel
        self.keyframes_only = keyframes_only
        self.token = token or CancellationToken()
        self.timestamps: list[float] = []

    def command(self) -> list[str]:
        """Return the ``ffmpeg`` argv used to sample this file."""
        width, height = self.size
        select = f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{self.interval_s:.6f})"
        command = [require_ffmpeg(), "-hide_banner", "-nostdin", "-v", "info"]
        # Both are decoder options, so they must precede the input they apply to.
        if self.hwaccel:
            command += ["-hwaccel", self.hwaccel]
        if self.keyframes_only:
            command += ["-skip_frame", "nokey"]
        if self.start_s > 0:
            command += ["-ss", f"{self.start_s:.3f}"]
        command += ["-i", str(self.path)]
        if self.duration_s is not None:
            command += ["-t", f"{self.duration_s:.3f}"]
        command += [
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"{select},scale={width}:{height},showinfo",
            *_passthrough_flag(),
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        return command

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield each sampled frame as a ``uint8`` array of shape ``(h, w)``."""
        width, height = self.size
        frame_bytes = width * height
        self.token.raise_if_cancelled()
        command = self.command()
        logger.debug("extracting frames: %s", " ".join(command))

        with tempfile.TemporaryFile() as error_sink:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=error_sink,
                stdin=subprocess.DEVNULL,
                bufsize=frame_bytes,
            )
            # Tracking hands ffmpeg to the token, which kills it on Ctrl+C. That
            # is what unblocks the read below; checking a flag alone would not.
            with self.token.track(process):
                try:
                    assert process.stdout is not None
                    while not self.token.cancelled:
                        payload = process.stdout.read(frame_bytes)
                        if len(payload) < frame_bytes:
                            break
                        yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width)
                finally:
                    with contextlib.suppress(OSError):
                        if process.stdout is not None:
                            process.stdout.close()
                    returncode = process.wait()
                    error_sink.seek(0)
                    log = error_sink.read().decode(errors="replace")

            # A killed decoder reports failure; say what actually happened.
            self.token.raise_if_cancelled()
            if returncode != 0:
                raise FrameExtractionError(f"ffmpeg failed for {self.path}: {log.strip()}")
            self.timestamps = [self.start_s + value for value in parse_showinfo_timestamps(log)]


#: Caches whether ``(method, codec)`` can be hardware-decoded on this machine.
#: GPU support is per-codec: a card may decode H.264 but have no MPEG-2 unit at
#: all, so a single global answer would be wrong.
_HWACCEL_SUPPORT: dict[tuple[str, str], bool] = {}
_HWACCEL_LOCK = threading.Lock()


def hwaccel_has_failed(method: str, codec: str) -> bool:
    """Return whether ``method`` is known not to work for ``codec`` in this run."""
    return _HWACCEL_SUPPORT.get((method, codec)) is False


def forget_hwaccel_failures() -> None:
    """Clear the cached hardware-decode probe results."""
    with _HWACCEL_LOCK:
        _HWACCEL_SUPPORT.clear()


def hwaccel_supported(method: str, path: Path, codec: str) -> bool:
    """Return whether ``method`` can decode ``codec``, probing once per run.

    Decoding a single frame is far cheaper than discovering the problem part
    way through a full-length file, and doing it under a lock means a machine
    with no GPU logs one warning rather than one per worker thread.

    Parameters
    ----------
    method
        ffmpeg ``-hwaccel`` method name.
    path
        A file using ``codec``, used as the probe subject.
    codec
        Codec name, which scopes the cached answer.

    Returns
    -------
    bool
        Whether hardware decoding should be attempted.
    """
    key = (method, codec)
    cached = _HWACCEL_SUPPORT.get(key)
    if cached is not None:
        return cached

    with _HWACCEL_LOCK:
        cached = _HWACCEL_SUPPORT.get(key)
        if cached is not None:
            return cached

        command = [
            require_ffmpeg(),
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-hwaccel",
            method,
            "-i",
            str(path),
            "-an",
            "-sn",
            "-dn",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, timeout=60, check=False)
            works = completed.returncode == 0
        except (OSError, subprocess.SubprocessError) as error:
            logger.debug("hwaccel probe for %s/%s errored: %s", method, codec, error)
            works = False

        if works:
            logger.info("using %s hardware decoding for %s", method, codec or "this codec")
        else:
            logger.warning(
                "hardware decoding (%s) is unavailable for %s; using software decoding",
                method,
                codec or "this codec",
            )
        _HWACCEL_SUPPORT[key] = works
        return works


def extract_frame_hashes(
    path: Path,
    *,
    start_s: float,
    duration_s: float | None,
    interval_s: float,
    size: tuple[int, int],
    hwaccel: str | None = None,
    keyframes_only: bool = False,
    codec: str = "",
    token: CancellationToken | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode ``path`` on a cadence and hash every sampled frame.

    When ``hwaccel`` is requested but unavailable, ffmpeg exits non-zero rather
    than degrading, so the decode is retried in software. Hardware support is
    per-codec, so the failure is remembered against this file's codec: a card
    that decodes H.264 but not MPEG-2 keeps its acceleration for the former.

    Parameters
    ----------
    hwaccel
        ffmpeg ``-hwaccel`` method, or ``None`` for software decoding.
    keyframes_only
        Decode only keyframes.
    codec
        Codec of the video stream, used to scope hardware-decode failures.

    Returns
    -------
    tuple of numpy.ndarray
        ``(timestamps, phashes, dhashes)``, aligned and equal in length.
    """
    token = token or CancellationToken()
    token.raise_if_cancelled()
    if hwaccel and not hwaccel_supported(hwaccel, path, codec):
        hwaccel = None

    try:
        stream, phashes, dhashes = _decode_and_hash(
            path,
            start_s=start_s,
            duration_s=duration_s,
            interval_s=interval_s,
            size=size,
            hwaccel=hwaccel,
            keyframes_only=keyframes_only,
            token=token,
        )
    except FrameExtractionError:
        if not hwaccel:
            raise
        # The probe passed but the full decode did not; distrust the device from
        # here rather than failing a file that software decoding can handle.
        with _HWACCEL_LOCK:
            _HWACCEL_SUPPORT[(hwaccel, codec)] = False
        logger.warning(
            "hardware decoding (%s) failed part way through %s; retrying in software",
            hwaccel,
            path.name,
        )
        stream, phashes, dhashes = _decode_and_hash(
            path,
            start_s=start_s,
            duration_s=duration_s,
            interval_s=interval_s,
            size=size,
            hwaccel=None,
            keyframes_only=keyframes_only,
            token=token,
        )

    timestamps = stream.timestamps
    if len(timestamps) != len(phashes):
        logger.warning(
            "ffmpeg reported %d timestamps for %d frames of %s; falling back to the "
            "nominal cadence",
            len(timestamps),
            len(phashes),
            path.name,
        )
        timestamps = [start_s + position * interval_s for position in range(len(phashes))]

    return (
        np.asarray(timestamps, dtype=np.float32),
        np.asarray(phashes, dtype=np.uint64),
        np.asarray(dhashes, dtype=np.uint64),
    )


def _decode_and_hash(
    path: Path,
    *,
    start_s: float,
    duration_s: float | None,
    interval_s: float,
    size: tuple[int, int],
    hwaccel: str | None,
    keyframes_only: bool,
    token: CancellationToken,
) -> tuple[FrameStream, list[int], list[int]]:
    """Run one decode pass, returning the stream and the hashes it produced."""
    stream = FrameStream(
        path,
        start_s=start_s,
        duration_s=duration_s,
        interval_s=interval_s,
        size=size,
        hwaccel=hwaccel,
        keyframes_only=keyframes_only,
        token=token,
    )
    phashes: list[int] = []
    dhashes: list[int] = []
    for frame in stream:
        hashes = hash_array(frame)
        phashes.append(hashes.phash)
        dhashes.append(hashes.dhash)
    return stream, phashes, dhashes


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
    completed = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL, check=False)
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
    """Builds frame-hash indexes, reusing cached ones whenever they are valid.

    ``hwaccel`` is deliberately not part of the index fingerprint. Video
    decoding is bit-exact by specification, so a hardware decoder returns the
    same pixels as a software one and the hashes are unchanged; making it a
    cache key would force a whole season to be re-indexed for no benefit the
    first time someone tries the flag.
    """

    def __init__(
        self,
        cache: FrameIndexCache,
        params: IndexParams,
        *,
        hwaccel: str | None = None,
        token: CancellationToken | None = None,
    ) -> None:
        self.cache = cache
        self.params = params
        self.hwaccel = hwaccel
        self.token = token or CancellationToken()

    def get(self, video: VideoFile, *, refresh: bool = False) -> FrameIndex:
        """Return the frame index for ``video``, building it only if needed.

        Parameters
        ----------
        video
            The probed video file.
        refresh
            Ignore any cached index and rebuild from the video.
        """
        self.token.raise_if_cancelled()
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
        info = probe_stream(video.path)
        size = _scaled_size((info.width, info.height), self.params.sample_width)
        span = video.duration_s - self.params.skip_head_s - self.params.skip_tail_s
        duration = span if span > 0 else None
        logger.info(
            "indexing %s (%.0fs, every %.2fs, %dx%d, %s%s%s)",
            video.name,
            video.duration_s,
            self.params.interval_s,
            *size,
            info.codec_name or "unknown codec",
            f", {self.hwaccel} decode" if self.hwaccel else "",
            ", keyframes only" if self.params.keyframes_only else "",
        )
        timestamps, phashes, dhashes = extract_frame_hashes(
            video.path,
            start_s=self.params.skip_head_s,
            duration_s=duration,
            interval_s=self.params.interval_s,
            size=size,
            hwaccel=self.hwaccel,
            keyframes_only=self.params.keyframes_only,
            codec=info.codec_name,
            token=self.token,
        )
        if timestamps.size == 0:
            raise FrameExtractionError(f"no frames could be decoded from {video.path}")
        logger.info("indexed %s: %d frames", video.name, timestamps.size)
        self._warn_if_too_sparse(video, timestamps)
        return FrameIndex(
            video_path=video.path,
            params=self.params,
            timestamps=timestamps,
            phashes=phashes,
            dhashes=dhashes,
            source_size=video.size_bytes,
            source_mtime_ns=video.mtime_ns,
        )

    def _warn_if_too_sparse(self, video: VideoFile, timestamps: np.ndarray) -> None:
        """Warn when keyframe-only decoding sampled far less than was asked for.

        Encoders vary wildly in how often they place keyframes. DVD MPEG-2 puts
        one roughly every half second, but a Blu-ray encode can go many seconds
        between them, and a still that falls in such a gap is simply invisible.
        Silently returning "unmatched" would look like the matcher failing
        rather than the sampling being too coarse to see anything.
        """
        if not self.params.keyframes_only or timestamps.size == 0:
            return
        covered = video.duration_s - self.params.skip_head_s - self.params.skip_tail_s
        if covered <= 0:
            return
        # Compare sample counts rather than the span between first and last
        # sample, so a file with a single keyframe reads as maximally sparse
        # instead of as a zero-length, apparently perfect span.
        wanted = max(1.0, covered / self.params.interval_s)
        achieved_interval = covered / timestamps.size
        if timestamps.size * _SPARSE_KEYFRAME_RATIO < wanted:
            logger.warning(
                "%s has keyframes only every %.1fs on average, far coarser than the "
                "requested %.1fs sampling; stills falling between them cannot be found. "
                "Consider dropping --keyframes-only, or adding --refine to re-check "
                "promising hits.",
                video.name,
                achieved_interval,
                self.params.interval_s,
            )

    def extract_window(
        self, video: VideoFile, *, start_s: float, end_s: float, interval_s: float
    ) -> FrameIndex:
        """Sample a short window of ``video`` densely, without touching the cache.

        This is the refinement half of the optional two-stage mode: index the
        whole file coarsely, then re-sample only around promising hits.
        """
        start = max(0.0, start_s)
        info = probe_stream(video.path)
        size = _scaled_size((info.width, info.height), self.params.sample_width)
        # Refinement deliberately decodes every frame: the point of the pass is
        # to see what a keyframe-limited or coarse first pass could not.
        timestamps, phashes, dhashes = extract_frame_hashes(
            video.path,
            start_s=start,
            duration_s=max(interval_s, end_s - start),
            interval_s=interval_s,
            size=size,
            hwaccel=self.hwaccel,
            codec=info.codec_name,
            token=self.token,
        )
        return FrameIndex(
            video_path=video.path,
            params=replace(
                self.params,
                interval_s=interval_s,
                skip_head_s=start,
                keyframes_only=False,
            ),
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

    token = indexer.token
    token.raise_if_cancelled()

    indexes: dict[Path, FrameIndex] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(videos))))
    try:
        futures = {pool.submit(indexer.get, video, refresh=refresh): video for video in videos}
        for future, video in futures.items():
            try:
                indexes[video.path] = future.result()
            except OperationCancelledError:
                logger.debug("indexing of %s was cancelled", video.name)
            except (FrameExtractionError, OSError) as error:
                logger.error("could not index %s: %s", video.name, error)
    finally:
        # Dropping queued work is the difference between stopping now and
        # decoding every remaining episode before noticing the interrupt.
        pool.shutdown(wait=True, cancel_futures=token.cancelled)

    token.raise_if_cancelled()
    return indexes
