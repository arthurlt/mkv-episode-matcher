"""``ffprobe`` inventory and the deliberately narrow duration filter.

Duration is used for exactly one thing in this tool: throwing away titles that
obviously are not episodes (disc menus, logo stings, trailers). It is never a
ranking or identity signal, because broadcast discs routinely report rounded,
copied, or outright identical runtimes for every episode of a season.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .models import SkipReason, VideoFile

__all__ = [
    "DurationFilter",
    "ProbeError",
    "find_mkv_files",
    "inventory",
    "probe_dimensions",
    "probe_duration",
    "probe_video",
]

logger = logging.getLogger(__name__)

#: Anything shorter than this is a menu, a sting, or a trailer, not an episode.
DEFAULT_MIN_DURATION_S = 5 * 60

_PROBE_TIMEOUT_S = 60


class ProbeError(RuntimeError):
    """Raised when ``ffprobe`` cannot report a duration for a file."""


def require_ffprobe() -> str:
    """Return the path to ``ffprobe``, raising if it is not installed."""
    executable = shutil.which("ffprobe")
    if executable is None:
        raise ProbeError("ffprobe was not found on PATH; install ffmpeg to continue")
    return executable


def find_mkv_files(directory: Path) -> list[Path]:
    """Return the ``.mkv`` files directly inside ``directory``, sorted by name.

    Sub-directories are ignored: a MakeMKV output folder is flat, and recursing
    risks pulling in unrelated discs.

    Parameters
    ----------
    directory
        Folder containing the rips.

    Returns
    -------
    list of pathlib.Path
        Sorted paths. Sorting is for stable reporting only -- disc/title order
        is never used as a matching prior.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".mkv"
    )


def probe_duration(path: Path) -> float:
    """Return the duration of ``path`` in seconds via ``ffprobe``.

    Parameters
    ----------
    path
        Video file to probe.

    Returns
    -------
    float
        Duration in seconds.

    Raises
    ------
    ProbeError
        If ``ffprobe`` fails or reports no usable duration.
    """
    command = [
        require_ffprobe(),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    logger.debug("probing duration: %s", " ".join(command))
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=False
    )
    if completed.returncode != 0:
        raise ProbeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")

    try:
        raw = json.loads(completed.stdout)["format"]["duration"]
        duration = float(raw)
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        raise ProbeError(f"ffprobe reported no duration for {path}") from error

    if duration <= 0:
        raise ProbeError(f"ffprobe reported a non-positive duration for {path}")
    return duration


def probe_dimensions(path: Path) -> tuple[int, int]:
    """Return the ``(width, height)`` of the first video stream in ``path``.

    The frame extractor needs the exact decoded frame size up front so it can
    read fixed-size records off a raw ``ffmpeg`` pipe.

    Raises
    ------
    ProbeError
        If the file has no readable video stream.
    """
    command = [
        require_ffprobe(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=False
    )
    if completed.returncode != 0:
        raise ProbeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as error:
        raise ProbeError(f"no video stream found in {path}") from error
    if width <= 0 or height <= 0:
        raise ProbeError(f"video stream in {path} has a degenerate size")
    return width, height


def probe_video(path: Path) -> VideoFile:
    """Probe ``path`` and capture the stat facts used for cache invalidation."""
    stat = path.stat()
    return VideoFile(
        path=path,
        duration_s=probe_duration(path),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def inventory(
    paths: Iterable[Path], *, on_error: Literal["raise", "zero"] = "raise"
) -> list[VideoFile]:
    """Probe every path, optionally degrading unreadable files to zero duration.

    Parameters
    ----------
    paths
        Video files to probe.
    on_error
        ``"raise"`` propagates :class:`ProbeError`; ``"zero"`` records a zero
        duration so the file surfaces in the report as skipped/unreadable
        rather than aborting the whole run.

    Returns
    -------
    list of VideoFile
        One entry per input path, in input order.
    """
    videos: list[VideoFile] = []
    for path in paths:
        try:
            videos.append(probe_video(path))
        except (ProbeError, OSError, subprocess.SubprocessError):
            if on_error == "raise":
                raise
            logger.warning("could not probe %s; treating as unreadable", path)
            stat = path.stat()
            videos.append(
                VideoFile(
                    path=path,
                    duration_s=0.0,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
    return videos


@dataclass(frozen=True, slots=True)
class DurationFilter:
    """A skip-floor over runtimes -- never a ranking signal.

    Attributes
    ----------
    min_duration_s
        Files shorter than this are skipped as menus/trailers.
    max_duration_s
        Optional ceiling for titles that clearly cannot be a single episode
        (a "Play All" playlist, say). ``None`` disables the ceiling, which is
        the default because v1 does not attempt to handle such titles anyway.

    Examples
    --------
    >>> from pathlib import Path
    >>> from mkv_episode_matcher.models import VideoFile
    >>> menu = VideoFile(Path("t04.mkv"), 42.0, 1, 1)
    >>> DurationFilter().classify(menu).value
    'too_short'
    """

    min_duration_s: float = DEFAULT_MIN_DURATION_S
    max_duration_s: float | None = None

    def classify(self, video: VideoFile) -> SkipReason | None:
        """Return why ``video`` should be skipped, or ``None`` to keep it."""
        if video.duration_s <= 0:
            return SkipReason.UNREADABLE
        if video.duration_s < self.min_duration_s:
            return SkipReason.TOO_SHORT
        if self.max_duration_s is not None and video.duration_s > self.max_duration_s:
            return SkipReason.TOO_LONG
        return None

    def partition(
        self, videos: Sequence[VideoFile]
    ) -> tuple[list[VideoFile], dict[Path, SkipReason]]:
        """Split ``videos`` into candidates and a map of skipped paths to reasons."""
        kept: list[VideoFile] = []
        skipped: dict[Path, SkipReason] = {}
        for video in videos:
            reason = self.classify(video)
            if reason is None:
                kept.append(video)
            else:
                logger.info("skipping %s (%s, %.1fs)", video.name, reason, video.duration_s)
                skipped[video.path] = reason
        return kept, skipped
