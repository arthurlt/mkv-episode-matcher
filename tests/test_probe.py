"""Tests for the ffprobe inventory layer and the duration skip-floor."""

from __future__ import annotations

from pathlib import Path

import pytest

from mkv_episode_matcher.models import SkipReason, VideoFile
from mkv_episode_matcher.probe import (
    DurationFilter,
    ProbeError,
    find_mkv_files,
    inventory,
    probe_duration,
    probe_video,
)

from .conftest import requires_ffmpeg


def make_video(duration_s: float) -> VideoFile:
    return VideoFile(Path("t00.mkv"), duration_s=duration_s, size_bytes=1, mtime_ns=1)


class TestDurationFilter:
    def test_skips_files_below_the_floor(self):
        assert DurationFilter().classify(make_video(120.0)) is SkipReason.TOO_SHORT

    def test_keeps_a_normal_episode(self):
        assert DurationFilter().classify(make_video(22 * 60)) is None

    def test_keeps_files_exactly_at_the_floor(self):
        assert DurationFilter(min_duration_s=300).classify(make_video(300.0)) is None

    def test_skips_absurdly_long_files(self):
        long_filter = DurationFilter(max_duration_s=2 * 60 * 60)

        assert long_filter.classify(make_video(5 * 60 * 60)) is SkipReason.TOO_LONG

    def test_no_upper_bound_by_default(self):
        assert DurationFilter().max_duration_s is None
        assert DurationFilter().classify(make_video(9 * 60 * 60)) is None

    def test_zero_duration_is_treated_as_unreadable(self):
        assert DurationFilter().classify(make_video(0.0)) is SkipReason.UNREADABLE

    def test_partitions_an_inventory(self):
        videos = [make_video(60.0), make_video(1320.0), make_video(1325.0)]

        kept, skipped = DurationFilter().partition(videos)

        assert [video.duration_s for video in kept] == [1320.0, 1325.0]
        assert skipped == {videos[0].path: SkipReason.TOO_SHORT}

    def test_identical_runtimes_are_never_used_to_rank(self):
        """Duration is a floor, not a ranking signal: equal runtimes all survive."""
        videos = [make_video(1320.0) for _ in range(5)]

        kept, skipped = DurationFilter().partition(videos)

        assert len(kept) == 5
        assert skipped == {}


class TestFindMkvFiles:
    def test_finds_mkv_files_in_sorted_order(self, tmp_path):
        for name in ("title_t02.mkv", "title_t00.mkv", "title_t01.mkv"):
            (tmp_path / name).touch()
        (tmp_path / "notes.txt").touch()

        assert [path.name for path in find_mkv_files(tmp_path)] == [
            "title_t00.mkv",
            "title_t01.mkv",
            "title_t02.mkv",
        ]

    def test_is_case_insensitive_about_the_extension(self, tmp_path):
        (tmp_path / "TITLE.MKV").touch()

        assert len(find_mkv_files(tmp_path)) == 1

    def test_ignores_subdirectories(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.mkv").touch()

        assert find_mkv_files(tmp_path) == []

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            find_mkv_files(tmp_path / "nope")


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestProbeAgainstRealFiles:
    def test_reads_the_duration_of_a_real_video(self, video_factory):
        path = video_factory("clip.mkv", [1, 2, 3], seconds_per_frame=1.0)

        assert probe_duration(path) == pytest.approx(3.0, abs=0.3)

    def test_probe_video_captures_cache_invalidation_facts(self, video_factory):
        path = video_factory("clip.mkv", [1, 2])

        video = probe_video(path)

        assert video.path == path
        assert video.size_bytes == path.stat().st_size
        assert video.mtime_ns == path.stat().st_mtime_ns

    def test_inventory_probes_every_file(self, video_factory, tmp_path):
        video_factory("a.mkv", [1, 2])
        video_factory("b.mkv", [3, 4])

        videos = inventory(find_mkv_files(tmp_path))

        assert [video.name for video in videos] == ["a.mkv", "b.mkv"]
        assert all(video.duration_s > 0 for video in videos)

    def test_a_non_video_file_raises_probe_error(self, tmp_path):
        broken = tmp_path / "broken.mkv"
        broken.write_bytes(b"this is not a matroska container")

        with pytest.raises(ProbeError):
            probe_duration(broken)

    def test_inventory_records_unreadable_files_as_zero_duration(self, tmp_path):
        broken = tmp_path / "broken.mkv"
        broken.write_bytes(b"nope")

        videos = inventory([broken], on_error="zero")

        assert videos[0].duration_s == 0.0
        assert DurationFilter().classify(videos[0]) is SkipReason.UNREADABLE
