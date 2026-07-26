"""Tests for dense frame sampling, hashing, and index cache invalidation."""

from __future__ import annotations

import numpy as np
import pytest

from mkv_episode_matcher.frames import (
    FrameIndex,
    FrameIndexCache,
    FrameIndexer,
    IndexParams,
    build_indexes,
)
from mkv_episode_matcher.normalize import hamming_distance, hash_array
from mkv_episode_matcher.probe import probe_video

from .conftest import make_pattern, requires_ffmpeg


def make_index(video, params: IndexParams, count: int = 3) -> FrameIndex:
    return FrameIndex(
        video_path=video.path,
        params=params,
        timestamps=np.arange(count, dtype=np.float32),
        phashes=np.arange(count, dtype=np.uint64),
        dhashes=np.arange(count, dtype=np.uint64),
        source_size=video.size_bytes,
        source_mtime_ns=video.mtime_ns,
    )


@pytest.fixture
def fake_video(tmp_path):
    from mkv_episode_matcher.models import VideoFile

    path = tmp_path / "title_t00.mkv"
    path.write_bytes(b"x" * 100)
    stat = path.stat()
    return VideoFile(path, 1320.0, stat.st_size, stat.st_mtime_ns)


class TestIndexParams:
    def test_fingerprint_is_stable(self):
        assert IndexParams().fingerprint == IndexParams().fingerprint

    def test_fingerprint_changes_with_the_interval(self):
        assert IndexParams(interval_s=1.0).fingerprint != IndexParams(interval_s=2.0).fingerprint

    def test_fingerprint_changes_with_the_sample_width(self):
        assert IndexParams().fingerprint != IndexParams(sample_width=640).fingerprint

    def test_rejects_a_non_positive_interval(self):
        with pytest.raises(ValueError, match="interval"):
            IndexParams(interval_s=0)


class TestFrameIndexPersistence:
    def test_round_trips_through_disk(self, fake_video, tmp_path):
        index = make_index(fake_video, IndexParams())
        target = tmp_path / "index.npz"

        index.save(target)
        loaded = FrameIndex.load(target)

        assert loaded.video_path == index.video_path
        assert loaded.params == index.params
        assert loaded.source_mtime_ns == index.source_mtime_ns
        np.testing.assert_array_equal(loaded.phashes, index.phashes)
        np.testing.assert_array_equal(loaded.timestamps, index.timestamps)

    def test_preserves_full_width_hashes(self, fake_video, tmp_path):
        index = make_index(fake_video, IndexParams())
        index.phashes[0] = np.uint64(0xFFFFFFFFFFFFFFFF)
        target = tmp_path / "index.npz"

        index.save(target)

        assert FrameIndex.load(target).phashes[0] == 0xFFFFFFFFFFFFFFFF

    def test_reports_its_length(self, fake_video):
        assert len(make_index(fake_video, IndexParams(), count=7)) == 7


class TestCacheInvalidation:
    def test_index_is_valid_for_an_unchanged_file(self, fake_video):
        assert make_index(fake_video, IndexParams()).is_valid_for(fake_video, IndexParams())

    def test_a_changed_mtime_invalidates_the_index(self, fake_video):
        from dataclasses import replace

        index = make_index(fake_video, IndexParams())

        assert not index.is_valid_for(replace(fake_video, mtime_ns=999), IndexParams())

    def test_a_changed_size_invalidates_the_index(self, fake_video):
        from dataclasses import replace

        index = make_index(fake_video, IndexParams())

        assert not index.is_valid_for(replace(fake_video, size_bytes=999), IndexParams())

    def test_changed_params_invalidate_the_index(self, fake_video):
        index = make_index(fake_video, IndexParams(interval_s=1.0))

        assert not index.is_valid_for(fake_video, IndexParams(interval_s=4.0))

    def test_cache_paths_are_unique_per_file(self, cache_dir, tmp_path):
        cache = FrameIndexCache(cache_dir)
        params = IndexParams()

        first = cache.path_for(tmp_path / "a" / "title_t00.mkv", params)
        second = cache.path_for(tmp_path / "b" / "title_t00.mkv", params)

        assert first != second

    def test_cache_paths_are_unique_per_params(self, cache_dir, tmp_path):
        cache = FrameIndexCache(cache_dir)
        path = tmp_path / "title_t00.mkv"

        assert cache.path_for(path, IndexParams(interval_s=1.0)) != cache.path_for(
            path, IndexParams(interval_s=2.0)
        )

    def test_store_then_load_returns_the_index(self, cache_dir, fake_video):
        cache = FrameIndexCache(cache_dir)
        params = IndexParams()
        index = make_index(fake_video, params)

        cache.store(index)

        assert cache.load(fake_video, params) is not None

    def test_load_returns_none_when_the_file_changed(self, cache_dir, fake_video):
        cache = FrameIndexCache(cache_dir)
        params = IndexParams()
        cache.store(make_index(fake_video, params))
        fake_video.path.write_bytes(b"y" * 250)
        changed = type(fake_video)(
            fake_video.path,
            fake_video.duration_s,
            fake_video.path.stat().st_size,
            fake_video.path.stat().st_mtime_ns,
        )

        assert cache.load(changed, params) is None

    def test_load_returns_none_for_a_corrupt_cache_entry(self, cache_dir, fake_video):
        cache = FrameIndexCache(cache_dir)
        params = IndexParams()
        cache.path_for(fake_video.path, params).write_bytes(b"garbage")

        assert cache.load(fake_video, params) is None


class TestNearest:
    def test_finds_the_closest_frame(self, fake_video):
        index = FrameIndex(
            video_path=fake_video.path,
            params=IndexParams(),
            timestamps=np.array([0.0, 10.0, 20.0], dtype=np.float32),
            phashes=np.array([0b0000, 0b0110, 0b1000], dtype=np.uint64),
            dhashes=np.zeros(3, dtype=np.uint64),
            source_size=1,
            source_mtime_ns=1,
        )

        position, distance = index.nearest(0b0111)

        assert position == 1
        assert distance == 1
        assert index.timestamps[position] == 10.0

    def test_empty_index_has_no_nearest_frame(self, fake_video):
        index = make_index(fake_video, IndexParams(), count=0)

        assert index.nearest(1) is None


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestBuildAgainstRealVideos:
    def test_index_covers_the_runtime_at_the_requested_cadence(self, video_factory, cache_dir):
        path = video_factory("clip.mkv", [1, 2, 3, 4, 5, 6], seconds_per_frame=1.0)
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=1.0))

        index = indexer.get(probe_video(path))

        assert 5 <= len(index) <= 7
        assert index.timestamps[0] == pytest.approx(0.0, abs=0.1)
        assert index.timestamps[-1] >= 4.0

    def test_planted_frame_is_found_at_the_right_timestamp(self, video_factory, cache_dir):
        seeds = [31, 32, 33, 34, 35, 36]
        path = video_factory("clip.mkv", seeds, seconds_per_frame=2.0)
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=1.0))
        index = indexer.get(probe_video(path))

        target = hash_array(make_pattern(34)).phash
        found = index.nearest(target)

        assert found is not None
        position, distance = found
        assert distance <= 6
        assert 6.0 <= index.timestamps[position] <= 8.5

    def test_a_coarser_interval_produces_fewer_frames(self, video_factory, cache_dir):
        path = video_factory("clip.mkv", list(range(41, 51)), seconds_per_frame=1.0)
        video = probe_video(path)
        cache = FrameIndexCache(cache_dir)

        dense = FrameIndexer(cache, IndexParams(interval_s=1.0)).get(video)
        coarse = FrameIndexer(cache, IndexParams(interval_s=5.0)).get(video)

        assert len(coarse) < len(dense)

    def test_second_run_reuses_the_cached_index(self, video_factory, cache_dir, monkeypatch):
        from mkv_episode_matcher import frames as frames_module

        path = video_factory("clip.mkv", [1, 2, 3, 4])
        video = probe_video(path)
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=1.0))
        indexer.get(video)

        calls = []
        original = frames_module.extract_frame_hashes

        def counting(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(frames_module, "extract_frame_hashes", counting)
        again = indexer.get(video)

        assert calls == []
        assert len(again) > 0

    def test_refresh_forces_a_rebuild(self, video_factory, cache_dir, monkeypatch):
        from mkv_episode_matcher import frames as frames_module

        path = video_factory("clip.mkv", [1, 2, 3, 4])
        video = probe_video(path)
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=1.0))
        indexer.get(video)

        calls = []
        original = frames_module.extract_frame_hashes
        monkeypatch.setattr(
            frames_module,
            "extract_frame_hashes",
            lambda *a, **k: (calls.append(a), original(*a, **k))[1],
        )
        indexer.get(video, refresh=True)

        assert len(calls) == 1

    def test_skip_head_offsets_the_timestamps(self, video_factory, cache_dir):
        path = video_factory("clip.mkv", list(range(61, 71)), seconds_per_frame=1.0)
        params = IndexParams(interval_s=1.0, skip_head_s=3.0)

        index = FrameIndexer(FrameIndexCache(cache_dir), params).get(probe_video(path))

        assert index.timestamps[0] == pytest.approx(3.0, abs=0.2)

    def test_build_indexes_handles_several_files_in_parallel(self, video_factory, cache_dir, tmp_path):
        video_factory("a.mkv", [1, 2, 3, 4])
        video_factory("b.mkv", [5, 6, 7, 8])
        videos = [probe_video(p) for p in sorted(tmp_path.glob("*.mkv"))]
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=1.0))

        indexes = build_indexes(indexer, videos, max_workers=2)

        assert set(indexes) == {video.path for video in videos}
        assert all(len(index) > 0 for index in indexes.values())

    def test_window_extraction_refines_around_a_hit(self, video_factory, cache_dir):
        seeds = [71, 72, 73, 74, 75, 76]
        path = video_factory("clip.mkv", seeds, seconds_per_frame=2.0)
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=4.0))
        video = probe_video(path)

        window = indexer.extract_window(video, start_s=4.0, end_s=8.0, interval_s=0.5)

        assert len(window) >= 4
        assert float(window.timestamps.min()) >= 3.9
        target = hash_array(make_pattern(73)).phash
        found = window.nearest(target)
        assert found is not None
        assert found[1] <= 6

    def test_distinct_videos_produce_distinct_indexes(self, video_factory, cache_dir, tmp_path):
        first = video_factory("a.mkv", [101, 102, 103, 104])
        second = video_factory("b.mkv", [201, 202, 203, 204])
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=1.0))

        index_a = indexer.get(probe_video(first))
        index_b = indexer.get(probe_video(second))

        target = hash_array(make_pattern(102)).phash
        best_a = index_a.nearest(target)
        best_b = index_b.nearest(target)

        assert best_a[1] < best_b[1]
        assert best_b[1] - best_a[1] >= 8

    def test_hashing_a_black_video_does_not_crash(self, tmp_path, cache_dir):
        from .conftest import encode_video

        black = [np.zeros((180, 320), dtype=np.uint8) for _ in range(6)]
        path = encode_video(tmp_path / "black.mkv", black)
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=1.0))

        assert len(indexer.get(probe_video(path))) > 0


def test_hamming_distance_between_planted_and_source_frame_is_small():
    """Guard the assumption the pipeline rests on: re-encoding preserves pHash."""
    pattern = make_pattern(99)
    noisy = np.clip(pattern.astype(np.int16) + 3, 0, 255).astype(np.uint8)

    assert hamming_distance(hash_array(pattern).phash, hash_array(noisy).phash) <= 2
