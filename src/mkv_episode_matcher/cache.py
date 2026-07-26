"""On-disk caches for provider JSON, downloaded stills, and frame indexes.

Everything the tool fetches or computes is cached under one root so that a
second run against the same disc is dominated by Hamming arithmetic rather than
network round-trips and video decoding.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .models import Still

__all__ = ["CacheRoot", "ImageCache", "JsonCache"]

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_IMAGE_TIMEOUT_S = 30.0


class CacheRoot:
    """The cache directory layout: API JSON, stills, and frame indexes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        for directory in (self.api_dir, self.stills_dir, self.frames_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> CacheRoot:
        """Return the cache root from ``MKV_MATCHER_CACHE`` or the XDG default."""
        override = os.environ.get("MKV_MATCHER_CACHE")
        if override:
            return cls(Path(override))
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return cls(base / "mkv-episode-matcher")

    @property
    def api_dir(self) -> Path:
        """Directory holding cached provider JSON responses."""
        return self.path / "api"

    @property
    def stills_dir(self) -> Path:
        """Directory holding downloaded still images."""
        return self.path / "stills"

    @property
    def frames_dir(self) -> Path:
        """Directory holding persisted per-file frame-hash indexes."""
        return self.path / "frames"

    def __repr__(self) -> str:
        """Return a readable representation naming the root path."""
        return f"CacheRoot({str(self.path)!r})"


class JsonCache:
    """A flat key/value cache of JSON documents with an optional TTL."""

    def __init__(self, directory: Path, *, ttl_s: float | None = None, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.ttl_s = ttl_s
        self.enabled = enabled
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        """Return the file backing ``key``.

        The key is slugified and suffixed with a digest, so keys containing
        slashes or traversal segments can never escape the cache directory.

        Examples
        --------
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     path = JsonCache(Path(directory)).path_for("tmdb/season/2")
        ...     path.suffix
        '.json'
        """
        slug = _UNSAFE.sub("-", key).strip("-")[:64] or "key"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.directory / f"{slug}-{digest}.json"

    def get(self, key: str) -> dict | None:
        """Return the cached document for ``key``, or ``None`` on a miss."""
        if not self.enabled:
            return None
        path = self.path_for(key)
        if not path.exists():
            return None
        if self.ttl_s is not None and (time.time() - path.stat().st_mtime) > self.ttl_s:
            logger.debug("cache entry %s expired", key)
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("discarding unreadable cache entry %s (%s)", key, error)
            return None

    def put(self, key: str, payload: dict) -> None:
        """Store ``payload`` under ``key``."""
        if not self.enabled:
            return
        path = self.path_for(key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(path)


class ImageCache:
    """Downloads provider stills once and keeps them on disk."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, still: Still) -> Path:
        """Return the on-disk location for ``still``."""
        return self.directory / still.filename

    def fetch(self, client: httpx.Client, still: Still) -> Path | None:
        """Return the local path of ``still``, downloading it if necessary.

        Returns
        -------
        pathlib.Path or None
            ``None`` when the image could not be retrieved; a missing still is
            a reason to have less evidence, never a reason to abort the run.
        """
        path = self.path_for(still)
        if path.exists() and path.stat().st_size > 0:
            return path

        try:
            response = client.get(still.url, timeout=_IMAGE_TIMEOUT_S, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as error:
            logger.warning("could not download still %s (%s)", still.url, error)
            return None

        if not response.content:
            logger.warning("still %s came back empty", still.url)
            return None

        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
        logger.debug("cached still %s -> %s", still.url, path.name)
        return path

    def fetch_many(
        self, client: httpx.Client, stills: Sequence[Still] | Iterable[Still], *, max_workers: int = 8
    ) -> dict[str, Path]:
        """Download several stills concurrently, keyed by URL.

        Stills that fail to download are simply absent from the result.
        """
        unique: dict[str, Still] = {still.url: still for still in stills}
        if not unique:
            return {}
        results: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(unique)))) as pool:
            futures = {
                pool.submit(self.fetch, client, still): url for url, still in unique.items()
            }
            for future, url in futures.items():
                path = future.result()
                if path is not None:
                    results[url] = path
        return results
