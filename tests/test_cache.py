"""Tests for the on-disk API/still caches."""

from __future__ import annotations

import os
import time

import httpx
import pytest

from mkv_episode_matcher.cache import CacheRoot, ImageCache, JsonCache
from mkv_episode_matcher.models import Still, StillKind


class TestJsonCache:
    def test_round_trips_a_payload(self, cache_dir):
        cache = JsonCache(cache_dir)

        cache.put("tmdb/season/2", {"episodes": [1, 2, 3]})

        assert cache.get("tmdb/season/2") == {"episodes": [1, 2, 3]}

    def test_missing_key_returns_none(self, cache_dir):
        assert JsonCache(cache_dir).get("nothing") is None

    def test_keys_with_slashes_do_not_escape_the_cache_directory(self, cache_dir):
        cache = JsonCache(cache_dir)

        cache.put("../../escape", {"a": 1})

        assert cache.get("../../escape") == {"a": 1}
        assert next(iter(cache_dir.rglob("*.json"))).is_relative_to(cache_dir)

    def test_corrupt_entry_is_treated_as_a_miss(self, cache_dir):
        cache = JsonCache(cache_dir)
        cache.put("key", {"a": 1})
        cache.path_for("key").write_text("{not json")

        assert cache.get("key") is None

    def test_entries_expire_once_past_the_ttl(self, cache_dir):
        cache = JsonCache(cache_dir, ttl_s=60)
        cache.put("key", {"a": 1})
        assert cache.get("key") == {"a": 1}

        aged = time.time() - 3600
        os.utime(cache.path_for("key"), (aged, aged))

        assert cache.get("key") is None

    def test_entries_within_the_ttl_survive(self, cache_dir):
        cache = JsonCache(cache_dir, ttl_s=3600)
        cache.put("key", {"a": 1})
        recent = time.time() - 60
        os.utime(cache.path_for("key"), (recent, recent))

        assert cache.get("key") == {"a": 1}

    def test_disabled_cache_never_stores_anything(self, cache_dir):
        cache = JsonCache(cache_dir, enabled=False)

        cache.put("key", {"a": 1})

        assert cache.get("key") is None
        assert list(cache_dir.iterdir()) == []


class TestImageCache:
    @staticmethod
    def still(url: str = "https://image.tmdb.org/t/p/w780/abc.jpg") -> Still:
        return Still(provider="tmdb", url=url, kind=StillKind.SCREENCAP)

    def test_downloads_and_stores_an_image(self, cache_dir):
        calls = []

        def handler(request):
            calls.append(request.url)
            return httpx.Response(200, content=b"\xff\xd8jpegbytes")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        cache = ImageCache(cache_dir)

        path = cache.fetch(client, self.still())

        assert path is not None
        assert path.read_bytes() == b"\xff\xd8jpegbytes"
        assert len(calls) == 1

    def test_second_fetch_hits_the_disk_cache(self, cache_dir):
        calls = []

        def handler(request):
            calls.append(request.url)
            return httpx.Response(200, content=b"data")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        cache = ImageCache(cache_dir)

        cache.fetch(client, self.still())
        cache.fetch(client, self.still())

        assert len(calls) == 1

    def test_a_failed_download_returns_none_and_leaves_no_file(self, cache_dir):
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
        cache = ImageCache(cache_dir)

        assert cache.fetch(client, self.still()) is None
        assert list(cache_dir.rglob("*.jpg")) == []

    def test_an_empty_response_is_not_cached(self, cache_dir):
        client = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b""))
        )

        assert ImageCache(cache_dir).fetch(client, self.still()) is None

    def test_distinct_urls_get_distinct_files(self, cache_dir):
        client = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x"))
        )
        cache = ImageCache(cache_dir)

        first = cache.fetch(client, self.still("https://img/a.jpg"))
        second = cache.fetch(client, self.still("https://img/b.jpg"))

        assert first != second

    def test_fetch_many_returns_a_path_per_successful_still(self, cache_dir):
        def handler(request):
            if request.url.path.endswith("missing.jpg"):
                return httpx.Response(404)
            return httpx.Response(200, content=b"x")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        stills = [
            self.still("https://img/a.jpg"),
            self.still("https://img/missing.jpg"),
            self.still("https://img/b.jpg"),
        ]

        paths = ImageCache(cache_dir).fetch_many(client, stills, max_workers=2)

        assert set(paths) == {"https://img/a.jpg", "https://img/b.jpg"}


class TestCacheRoot:
    def test_creates_the_expected_layout(self, tmp_path):
        root = CacheRoot(tmp_path / "cache")

        assert root.api_dir.is_dir()
        assert root.stills_dir.is_dir()
        assert root.frames_dir.is_dir()

    def test_directories_are_nested_under_the_root(self, tmp_path):
        root = CacheRoot(tmp_path / "cache")

        assert root.frames_dir.parent == root.path

    def test_default_root_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MKV_MATCHER_CACHE", str(tmp_path / "from-env"))

        assert CacheRoot.default().path == tmp_path / "from-env"


def test_still_cache_filenames_are_stable_and_safe():
    still = Still(provider="tvdb", url="https://artworks.thetvdb.com/banners/x.jpg?q=1")

    assert still.filename.endswith(".jpg")
    assert "/" not in still.filename
    assert still.filename == Still(provider="tvdb", url=still.url).filename


def test_still_filename_falls_back_to_jpg_for_extensionless_urls():
    assert Still(provider="tmdb", url="https://img/abc").filename.endswith(".jpg")


@pytest.mark.parametrize("status", [500, 503])
def test_image_fetch_gives_up_on_server_errors(cache_dir, status):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status)))

    assert ImageCache(cache_dir).fetch(client, Still("tmdb", "https://img/a.jpg")) is None
