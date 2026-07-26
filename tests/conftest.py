"""Shared pytest fixtures for the MKV episode matcher test-suite."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
API_FIXTURE_ROOT = FIXTURE_ROOT / "api"


def ffmpeg_available() -> bool:
    """Return ``True`` when both ``ffmpeg`` and ``ffprobe`` are on ``PATH``."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg/ffprobe not installed"
)


def make_pattern(seed: int, width: int = 320, height: int = 180) -> np.ndarray:
    """Build a deterministic, visually distinctive grayscale pattern.

    Each ``seed`` produces a different arrangement of large blocks and ramps, so
    that two patterns with different seeds are far apart under a perceptual hash
    while remaining stable under rescaling and re-encoding.

    Parameters
    ----------
    seed
        Chooses the pattern. Different seeds give visually different images.
    width, height
        Output size in pixels.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(height, width)``.

    Examples
    --------
    >>> make_pattern(1, 8, 4).shape
    (4, 8)
    >>> bool((make_pattern(1, 32, 32) != make_pattern(2, 32, 32)).any())
    True
    """
    rng = np.random.default_rng(seed)
    blocks_y, blocks_x = 6, 8
    coarse = rng.integers(0, 256, size=(blocks_y, blocks_x), dtype=np.uint16)
    image = np.kron(coarse, np.ones((height // blocks_y + 1, width // blocks_x + 1)))
    image = image[:height, :width]
    ramp = np.linspace(0, 60, width, dtype=np.float64)[None, :]
    return np.clip(image + ramp, 0, 255).astype(np.uint8)


@pytest.fixture
def pattern_image() -> Image.Image:
    """Return a deterministic RGB test image."""
    return Image.fromarray(make_pattern(7)).convert("RGB")


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Return an empty on-disk cache directory."""
    path = tmp_path / "cache"
    path.mkdir()
    return path


@pytest.fixture
def api_fixture() -> "ApiFixtureLoader":
    """Return a loader for recorded provider API responses."""
    return ApiFixtureLoader(API_FIXTURE_ROOT)


class ApiFixtureLoader:
    """Load recorded JSON API responses from ``tests/fixtures/api``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, name: str) -> dict:
        """Return the parsed JSON fixture called ``name`` (without extension)."""
        return json.loads((self.root / f"{name}.json").read_text())


def encode_video(
    destination: Path,
    frames: list[np.ndarray],
    *,
    seconds_per_frame: float = 1.0,
    fps: int = 10,
) -> Path:
    """Encode ``frames`` into a small MKV, holding each frame for a fixed time.

    Parameters
    ----------
    destination
        Path of the ``.mkv`` file to write.
    frames
        Grayscale ``uint8`` arrays, all the same shape.
    seconds_per_frame
        How long each supplied frame is held on screen.
    fps
        Output frame rate of the encoded video.

    Returns
    -------
    pathlib.Path
        The ``destination`` path.
    """
    height, width = frames[0].shape
    repeats = max(1, round(seconds_per_frame * fps))
    raw = b"".join(bytes(frame) for frame in frames for _ in range(repeats))
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    subprocess.run(command, input=raw, check=True, capture_output=True)
    return destination


@pytest.fixture
def video_factory(tmp_path: Path) -> Iterator[object]:
    """Return a callable that encodes deterministic test videos into ``tmp_path``."""

    def factory(
        name: str,
        seeds: list[int],
        *,
        seconds_per_frame: float = 1.0,
        directory: Path | None = None,
    ) -> Path:
        target_dir = directory or tmp_path
        target_dir.mkdir(parents=True, exist_ok=True)
        frames = [make_pattern(seed) for seed in seeds]
        return encode_video(
            target_dir / name, frames, seconds_per_frame=seconds_per_frame
        )

    yield factory
