"""Shared pytest fixtures for the MKV episode matcher test-suite."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import httpx
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


class RecordedProviderApi:
    """An ``httpx`` transport replaying recorded TMDB/TheTVDB responses.

    Real ``httpx`` request building, redirects, and error handling still run;
    only the socket is replaced. Every request is recorded so tests can assert
    that the disk caches actually prevented repeat calls.
    """

    def __init__(self, root: Path = API_FIXTURE_ROOT) -> None:
        self.root = root
        self.requests: list[httpx.Request] = []
        self.image_bytes: dict[str, bytes] = {}
        self.default_image = _tiny_jpeg()

    @property
    def transport(self) -> httpx.MockTransport:
        """Return a transport that can be handed to :class:`httpx.Client`."""
        return httpx.MockTransport(self.handle)

    def client(self) -> httpx.Client:
        """Return an ``httpx`` client wired to this recorded API."""
        return httpx.Client(transport=self.transport)

    def paths_called(self, needle: str) -> int:
        """Return how many recorded requests contain ``needle`` in their URL."""
        return sum(1 for request in self.requests if needle in str(request.url))

    def _json(self, name: str) -> httpx.Response:
        path = self.root / f"{name}.json"
        if not path.exists():
            return httpx.Response(404, json={"status": "failure", "message": "not found"})
        return httpx.Response(200, json=json.loads(path.read_text()))

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Route a request to a recorded fixture."""
        self.requests.append(request)
        host, path = request.url.host, request.url.path

        if host == "api.themoviedb.org":
            return self._handle_tmdb(path)
        if host == "api4.thetvdb.com":
            return self._handle_tvdb(path)
        if host in {"image.tmdb.org", "artworks.thetvdb.com"}:
            key = str(request.url)
            return httpx.Response(200, content=self.image_bytes.get(key, self.default_image))
        return httpx.Response(404, json={"message": f"unrouted host {host}"})

    def _handle_tmdb(self, path: str) -> httpx.Response:
        parts = path.strip("/").split("/")
        if parts[:2] == ["3", "search"]:
            return self._json("tmdb_search")
        if len(parts) == 8 and parts[3] == "season" and parts[7] == "images":
            return self._json(f"tmdb_images_s{int(parts[4]):02d}e{int(parts[6]):02d}")
        if len(parts) == 5 and parts[3] == "season":
            return self._json(f"tmdb_season_{int(parts[4])}")
        return httpx.Response(404, json={"status_message": f"unrouted tmdb path {path}"})

    def _handle_tvdb(self, path: str) -> httpx.Response:
        parts = path.strip("/").split("/")
        if parts[-1] == "login":
            return self._json("tvdb_login")
        if parts[1] == "search":
            return self._json("tvdb_search")
        if parts[1] == "series" and "episodes" in parts:
            return self._json("tvdb_episodes_s2")
        if parts[1] == "episodes" and parts[-1] == "extended":
            return self._json(f"tvdb_episode_extended_{parts[2]}")
        return httpx.Response(404, json={"status": "failure", "message": f"unrouted {path}"})


def _tiny_jpeg() -> bytes:
    """Return the bytes of a small valid JPEG, used as a stand-in still."""
    buffer = io.BytesIO()
    Image.fromarray(make_pattern(1, 128, 72)).save(buffer, format="JPEG")
    return buffer.getvalue()


def still_bytes(seed: int, width: int = 320, height: int = 180) -> bytes:
    """Return JPEG bytes of the deterministic pattern for ``seed``."""
    buffer = io.BytesIO()
    Image.fromarray(make_pattern(seed, width, height)).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


@pytest.fixture
def recorded_api() -> RecordedProviderApi:
    """Return a recorded TMDB/TheTVDB API backed by JSON fixtures."""
    return RecordedProviderApi()


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
