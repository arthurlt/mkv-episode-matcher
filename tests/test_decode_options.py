"""Tests for hardware-accelerated and keyframe-only decoding.

Decode is 90%+ of indexing time, so these are the two levers that matter. Both
must be safe: a missing GPU has to degrade to software rather than fail, and
neither may change the hashes that a cached index is compared against.
"""

from __future__ import annotations

import numpy as np
import pytest

from mkv_episode_matcher.frames import (
    FrameIndexCache,
    FrameIndexer,
    FrameStream,
    IndexParams,
    forget_hwaccel_failures,
    hwaccel_has_failed,
)
from mkv_episode_matcher.probe import probe_stream, probe_video

from .conftest import requires_ffmpeg


@pytest.fixture(autouse=True)
def _clean_hwaccel_state():
    """Keep the per-run hwaccel failure memory from leaking between tests."""
    forget_hwaccel_failures()
    yield
    forget_hwaccel_failures()


def stream_for(path, **kwargs) -> FrameStream:
    defaults = {
        "start_s": 0.0,
        "duration_s": None,
        "interval_s": 1.0,
        "size": (320, 180),
    }
    return FrameStream(path, **(defaults | kwargs))


class TestCommandConstruction:
    def test_no_decoder_flags_by_default(self, tmp_path):
        command = stream_for(tmp_path / "a.mkv").command()

        assert "-hwaccel" not in command
        assert "-skip_frame" not in command

    def test_hwaccel_is_passed_as_an_input_option(self, tmp_path):
        command = stream_for(tmp_path / "a.mkv", hwaccel="vaapi").command()

        assert command[command.index("-hwaccel") + 1] == "vaapi"
        assert command.index("-hwaccel") < command.index("-i")

    def test_keyframe_skipping_is_passed_as_an_input_option(self, tmp_path):
        command = stream_for(tmp_path / "a.mkv", keyframes_only=True).command()

        assert command[command.index("-skip_frame") + 1] == "nokey"
        assert command.index("-skip_frame") < command.index("-i")

    def test_both_options_can_apply_at_once(self, tmp_path):
        command = stream_for(tmp_path / "a.mkv", hwaccel="cuda", keyframes_only=True).command()

        assert "-hwaccel" in command
        assert "-skip_frame" in command

    def test_the_select_filter_is_unchanged_by_hardware_decoding(self, tmp_path):
        software = stream_for(tmp_path / "a.mkv").command()
        accelerated = stream_for(tmp_path / "a.mkv", hwaccel="cuda").command()

        assert software[software.index("-vf") + 1] == accelerated[accelerated.index("-vf") + 1]


class TestFingerprinting:
    def test_keyframe_only_sampling_gets_its_own_cache_entry(self):
        """It changes which frames exist, so it must not reuse a full-decode index."""
        assert IndexParams().fingerprint != IndexParams(keyframes_only=True).fingerprint

    def test_the_cache_path_separates_the_two_sampling_modes(self, cache_dir, tmp_path):
        cache = FrameIndexCache(cache_dir)
        path = tmp_path / "title_t00.mkv"

        assert cache.path_for(path, IndexParams()) != cache.path_for(
            path, IndexParams(keyframes_only=True)
        )

    def test_hwaccel_does_not_invalidate_a_cached_index(self, cache_dir, tmp_path):
        """Decoders are bit-exact per spec, so the same frames hash the same either way."""
        cache = FrameIndexCache(cache_dir)
        path = tmp_path / "title_t00.mkv"
        indexer = FrameIndexer(cache, IndexParams(), hwaccel="cuda")

        assert indexer.params.fingerprint == IndexParams().fingerprint
        assert cache.path_for(path, indexer.params) == cache.path_for(path, IndexParams())


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestHardwareFallback:
    def test_an_unavailable_hwaccel_falls_back_to_software(self, video_factory, cache_dir):
        """This VM has no GPU: ffmpeg exits non-zero, and we must still index."""
        path = video_factory("clip.mkv", [1, 2, 3, 4, 5, 6])
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(), hwaccel="vaapi")

        index = indexer.get(probe_video(path))

        assert len(index) >= 5

    def test_the_fallback_produces_the_same_hashes_as_plain_software(
        self, video_factory, cache_dir
    ):
        path = video_factory("clip.mkv", [11, 12, 13, 14, 15, 16])
        video = probe_video(path)
        cache = FrameIndexCache(cache_dir)

        software = FrameIndexer(cache, IndexParams()).get(video)
        fell_back = FrameIndexer(cache, IndexParams(), hwaccel="cuda").get(video, refresh=True)

        np.testing.assert_array_equal(software.phashes, fell_back.phashes)
        np.testing.assert_array_equal(software.timestamps, fell_back.timestamps)

    def test_a_failed_hwaccel_is_remembered_for_the_rest_of_the_run(self, video_factory, cache_dir):
        """Retrying a missing GPU once per file would waste a process launch each time."""
        first = video_factory("a.mkv", [21, 22, 23, 24])
        second = video_factory("b.mkv", [25, 26, 27, 28])
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(), hwaccel="vaapi")
        codec = probe_stream(first).codec_name

        indexer.get(probe_video(first))
        assert hwaccel_has_failed("vaapi", codec)

        commands = []
        original = FrameStream.command

        def spy(self):
            command = original(self)
            commands.append(command)
            return command

        FrameStream.command = spy
        try:
            indexer.get(probe_video(second))
        finally:
            FrameStream.command = original

        assert commands, "the second file should still have been decoded"
        assert not any("-hwaccel" in command for command in commands)

    def test_hwaccel_auto_matches_software_exactly(self, video_factory, cache_dir):
        """``auto`` uses a GPU where one exists, so this is a cross-decoder check there."""
        path = video_factory("clip.mkv", [31, 32, 33, 34, 35, 36])
        video = probe_video(path)
        cache = FrameIndexCache(cache_dir)

        software = FrameIndexer(cache, IndexParams()).get(video)
        automatic = FrameIndexer(cache, IndexParams(), hwaccel="auto").get(video, refresh=True)

        np.testing.assert_array_equal(software.phashes, automatic.phashes)

    def test_a_nonsense_hwaccel_name_still_yields_an_index(self, video_factory, cache_dir):
        path = video_factory("clip.mkv", [41, 42, 43, 44])
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(), hwaccel="notathing")

        assert len(indexer.get(probe_video(path))) > 0


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestProbing:
    def test_a_working_method_is_probed_only_once(self, video_factory, cache_dir):
        first = video_factory("a.mkv", [71, 72, 73, 74])
        second = video_factory("b.mkv", [75, 76, 77, 78])
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(), hwaccel="vaapi")

        indexer.get(probe_video(first))
        indexer.get(probe_video(second))

        assert hwaccel_has_failed("vaapi", probe_stream(first).codec_name)

    def test_indexing_many_files_warns_about_a_missing_gpu_only_once(
        self, video_factory, cache_dir, caplog, tmp_path
    ):
        """Four parallel workers must not produce four identical warnings."""
        from mkv_episode_matcher.frames import build_indexes

        rips = tmp_path / "rips"
        videos = [
            probe_video(video_factory(f"t{n}.mkv", [80 + n, 81 + n, 82 + n], directory=rips))
            for n in range(4)
        ]
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(), hwaccel="vaapi")

        with caplog.at_level("WARNING"):
            indexes = build_indexes(indexer, videos, max_workers=4)

        assert len(indexes) == 4
        assert caplog.text.lower().count("hardware decoding") == 1


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestKeyframeOnly:
    def test_it_produces_a_usable_index(self, video_factory, cache_dir):
        path = video_factory("clip.mkv", [51, 52, 53, 54, 55, 56], seconds_per_frame=1.0)
        params = IndexParams(interval_s=1.0, keyframes_only=True)

        index = FrameIndexer(FrameIndexCache(cache_dir), params).get(probe_video(path))

        assert len(index) > 0

    def test_a_dense_gop_still_samples_at_the_requested_cadence(self, video_factory, cache_dir):
        """DVD-like encodes keyframe every half second, so nothing is lost."""
        path = video_factory(
            "dense.mkv", list(range(90, 100)), seconds_per_frame=1.0, keyframe_interval_s=0.5
        )
        params = IndexParams(interval_s=1.0, keyframes_only=True)

        index = FrameIndexer(FrameIndexCache(cache_dir), params).get(probe_video(path))

        gaps = np.diff(index.timestamps)
        assert len(index) >= 9
        assert float(gaps.max()) <= 1.5

    def test_sparse_keyframes_are_called_out_rather_than_silently_missing_stills(
        self, video_factory, cache_dir, caplog
    ):
        path = video_factory(
            "sparse.mkv", list(range(110, 120)), seconds_per_frame=1.0, keyframe_interval_s=5.0
        )
        params = IndexParams(interval_s=1.0, keyframes_only=True)

        with caplog.at_level("WARNING"):
            FrameIndexer(FrameIndexCache(cache_dir), params).get(probe_video(path))

        assert "keyframes only every" in caplog.text
        assert "--refine" in caplog.text

    def test_a_file_with_a_single_keyframe_is_the_loudest_case(
        self, video_factory, cache_dir, caplog
    ):
        """One keyframe over the whole file is maximally sparse, not a perfect span."""
        path = video_factory("onekey.mkv", list(range(140, 150)), seconds_per_frame=2.0)
        params = IndexParams(interval_s=1.0, keyframes_only=True)

        with caplog.at_level("WARNING"):
            index = FrameIndexer(FrameIndexCache(cache_dir), params).get(probe_video(path))

        assert len(index) == 1
        assert "keyframes only every" in caplog.text

    def test_a_dense_gop_does_not_trigger_the_warning(self, video_factory, cache_dir, caplog):
        path = video_factory(
            "dense.mkv", list(range(120, 130)), seconds_per_frame=1.0, keyframe_interval_s=0.5
        )
        params = IndexParams(interval_s=1.0, keyframes_only=True)

        with caplog.at_level("WARNING"):
            FrameIndexer(FrameIndexCache(cache_dir), params).get(probe_video(path))

        assert "keyframes only every" not in caplog.text

    def test_full_decode_never_triggers_the_warning(self, video_factory, cache_dir, caplog):
        path = video_factory("clip.mkv", list(range(130, 136)), seconds_per_frame=2.0)

        with caplog.at_level("WARNING"):
            FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=1.0)).get(
                probe_video(path)
            )

        assert "keyframes only every" not in caplog.text

    def test_timestamps_still_point_at_the_real_content(self, video_factory, cache_dir):
        """Keyframe sampling must not reintroduce the fps-filter timestamp drift."""
        from mkv_episode_matcher.normalize import hamming_distance, hash_array

        from .conftest import make_pattern

        seeds = [600 + slot for slot in range(10)]
        seconds_per_slot = 2.0
        path = video_factory("clip.mkv", seeds, seconds_per_frame=seconds_per_slot)
        reference = [hash_array(make_pattern(seed)).phash for seed in seeds]
        params = IndexParams(interval_s=1.0, keyframes_only=True)

        index = FrameIndexer(FrameIndexCache(cache_dir), params).get(probe_video(path))

        for timestamp, phash in zip(index.timestamps, index.phashes, strict=True):
            expected = min(int(float(timestamp) // seconds_per_slot), len(seeds) - 1)
            distances = [hamming_distance(int(phash), value) for value in reference]

            assert distances.index(min(distances)) == expected

    def test_frames_it_does_sample_hash_identically_to_a_full_decode(
        self, video_factory, cache_dir
    ):
        """Skipping frames changes which are sampled, never how they are hashed."""
        path = video_factory("clip.mkv", list(range(61, 71)), seconds_per_frame=2.0)
        video = probe_video(path)
        cache = FrameIndexCache(cache_dir)

        full = FrameIndexer(cache, IndexParams(interval_s=1.0)).get(video)
        keyframe = FrameIndexer(cache, IndexParams(interval_s=1.0, keyframes_only=True)).get(video)

        full_by_time = dict(zip(full.timestamps.tolist(), full.phashes.tolist(), strict=True))
        shared = [
            (time, value)
            for time, value in zip(
                keyframe.timestamps.tolist(), keyframe.phashes.tolist(), strict=True
            )
            if time in full_by_time
        ]
        assert shared, "the two modes should sample at least one common timestamp"
        assert all(full_by_time[time] == value for time, value in shared)


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestProbeStream:
    def test_reports_the_codec_alongside_the_size(self, video_factory):
        info = probe_stream(video_factory("clip.mkv", [1, 2]))

        assert info.codec_name == "h264"
        assert (info.width, info.height) == (320, 180)

    def test_missing_files_raise(self, tmp_path):
        from mkv_episode_matcher.probe import ProbeError

        with pytest.raises(ProbeError):
            probe_stream(tmp_path / "nope.mkv")
