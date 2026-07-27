"""Tests for cooperative cancellation of the worker threads.

Indexing a season means minutes of ``ffmpeg`` spread over a thread pool. A
thread cannot be killed from outside, so Ctrl+C has to work by consent: set a
flag, kill the child processes the workers are blocked on, and refuse to start
anything new. Anything less leaves the user staring at a terminal that ignores
them and, worse, orphaned decoders still burning CPU after the process exits.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mkv_episode_matcher.cancellation import (
    CancellationToken,
    OperationCancelledError,
    cancel_on_sigint,
)
from mkv_episode_matcher.frames import (
    FrameIndexCache,
    FrameIndexer,
    FrameStream,
    IndexParams,
    build_indexes,
)
from mkv_episode_matcher.probe import probe_video

from .conftest import requires_ffmpeg


def _raise_second_interrupt() -> None:
    """Send a second SIGINT and give the default handler a moment to fire."""
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(0.1)


def sleeper() -> subprocess.Popen:
    """Start a child process that will outlive the test unless it is killed."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class TestCancellationToken:
    def test_starts_uncancelled(self):
        assert CancellationToken().cancelled is False

    def test_cancel_sets_the_flag(self):
        token = CancellationToken()

        token.cancel()

        assert token.cancelled is True

    def test_raise_if_cancelled_is_quiet_until_cancelled(self):
        token = CancellationToken()
        token.raise_if_cancelled()

        token.cancel()

        with pytest.raises(OperationCancelledError):
            token.raise_if_cancelled()

    def test_cancelling_twice_is_harmless(self):
        token = CancellationToken()

        token.cancel()
        token.cancel()

        assert token.cancelled is True

    def test_callbacks_run_on_cancellation(self):
        token = CancellationToken()
        called = []
        token.on_cancel(lambda: called.append(True))

        token.cancel()

        assert called == [True]

    def test_a_callback_registered_after_cancellation_runs_immediately(self):
        token = CancellationToken()
        token.cancel()
        called = []

        token.on_cancel(lambda: called.append(True))

        assert called == [True]

    def test_a_failing_callback_does_not_stop_the_others(self):
        token = CancellationToken()
        called = []

        def boom() -> None:
            raise RuntimeError("callback exploded")

        token.on_cancel(boom)
        token.on_cancel(lambda: called.append(True))

        token.cancel()

        assert called == [True]


class TestProcessTracking:
    def test_cancelling_kills_a_tracked_process(self):
        token = CancellationToken()
        process = sleeper()

        def body() -> None:
            with token.track(process):
                process.wait()

        worker = threading.Thread(target=body)
        worker.start()
        time.sleep(0.2)
        token.cancel()
        worker.join(timeout=10)

        assert not worker.is_alive()
        assert process.poll() is not None

    def test_a_process_started_after_cancellation_is_killed_at_once(self):
        token = CancellationToken()
        token.cancel()
        process = sleeper()

        with pytest.raises(OperationCancelledError), token.track(process):
            pytest.fail("the body must not run")

        process.wait(timeout=10)
        assert process.poll() is not None

    def test_a_finished_process_is_no_longer_tracked(self):
        token = CancellationToken()
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        with token.track(process):
            process.wait()

        assert token.tracked_process_count == 0

    def test_tracking_is_released_even_when_the_body_raises(self):
        token = CancellationToken()
        process = subprocess.Popen([sys.executable, "-c", "pass"])

        with pytest.raises(ValueError, match="boom"), token.track(process):
            raise ValueError("boom")

        process.wait(timeout=10)
        assert token.tracked_process_count == 0

    def test_many_threads_can_track_and_cancel_at_once(self):
        token = CancellationToken()
        processes = [sleeper() for _ in range(6)]
        ready = threading.Barrier(len(processes) + 1)

        def body(process: subprocess.Popen) -> None:
            try:
                with token.track(process):
                    ready.wait(timeout=10)
                    process.wait()
            except OperationCancelledError:
                ready.wait(timeout=10)

        workers = [threading.Thread(target=body, args=(p,)) for p in processes]
        for worker in workers:
            worker.start()
        ready.wait(timeout=10)
        token.cancel()
        for worker in workers:
            worker.join(timeout=10)

        assert all(process.poll() is not None for process in processes)
        assert token.tracked_process_count == 0


class TestSigintHandling:
    def test_sigint_cancels_the_token(self):
        token = CancellationToken()

        with cancel_on_sigint(token):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.1)

        assert token.cancelled is True

    def test_the_previous_handler_is_restored_on_exit(self):
        previous = signal.getsignal(signal.SIGINT)

        with cancel_on_sigint(CancellationToken()):
            assert signal.getsignal(signal.SIGINT) is not previous

        assert signal.getsignal(signal.SIGINT) is previous

    def test_a_second_sigint_is_left_to_the_default_handler(self):
        """One Ctrl+C asks nicely; a second must always be able to abort."""
        token = CancellationToken()

        with cancel_on_sigint(token):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.1)
            assert token.cancelled

            with pytest.raises(KeyboardInterrupt):
                _raise_second_interrupt()

    def test_it_is_a_no_op_off_the_main_thread(self):
        """Only the main thread may install handlers; workers must not crash."""
        token = CancellationToken()
        errors = []

        def body() -> None:
            try:
                with cancel_on_sigint(token):
                    pass
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=body)
        worker.start()
        worker.join(timeout=10)

        assert errors == []


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestFrameStreamCancellation:
    def test_iteration_stops_and_raises_once_cancelled(self, video_factory):
        path = video_factory("clip.mkv", list(range(200, 220)), seconds_per_frame=0.5)
        token = CancellationToken()
        stream = FrameStream(
            path,
            start_s=0.0,
            duration_s=None,
            interval_s=0.5,
            size=(320, 180),
            token=token,
        )

        iterator = iter(stream)
        next(iterator)
        token.cancel()

        with pytest.raises(OperationCancelledError):
            for _ in iterator:
                pass

    def test_the_ffmpeg_child_is_dead_afterwards(self, video_factory):
        path = video_factory("clip.mkv", list(range(220, 240)), seconds_per_frame=0.5)
        token = CancellationToken()
        stream = FrameStream(
            path, start_s=0.0, duration_s=None, interval_s=0.5, size=(320, 180), token=token
        )

        iterator = iter(stream)
        next(iterator)
        token.cancel()
        with pytest.raises(OperationCancelledError):
            for _ in iterator:
                pass

        assert token.tracked_process_count == 0

    def test_an_already_cancelled_token_never_launches_ffmpeg(self, video_factory):
        path = video_factory("clip.mkv", [1, 2, 3, 4])
        token = CancellationToken()
        token.cancel()
        stream = FrameStream(
            path, start_s=0.0, duration_s=None, interval_s=1.0, size=(320, 180), token=token
        )

        with pytest.raises(OperationCancelledError):
            list(stream)


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestIndexerCancellation:
    def test_a_cancelled_indexer_does_not_build(self, video_factory, cache_dir):
        path = video_factory("clip.mkv", [1, 2, 3, 4])
        token = CancellationToken()
        token.cancel()
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(), token=token)

        with pytest.raises(OperationCancelledError):
            indexer.get(probe_video(path))

    def test_no_partial_index_is_left_in_the_cache(self, video_factory, cache_dir):
        """A truncated index that looked valid would silently break later runs."""
        path = video_factory("clip.mkv", list(range(240, 260)), seconds_per_frame=0.5)
        token = CancellationToken()
        cache = FrameIndexCache(cache_dir)
        indexer = FrameIndexer(cache, IndexParams(interval_s=0.5), token=token)
        video = probe_video(path)

        def cancel_soon() -> None:
            time.sleep(0.05)
            token.cancel()

        threading.Thread(target=cancel_soon).start()
        with pytest.raises(OperationCancelledError):
            indexer.get(video)

        assert cache.load(video, IndexParams(interval_s=0.5)) is None

    def test_a_cached_index_is_still_served_after_cancellation(self, video_factory, cache_dir):
        """Work already finished must not be thrown away."""
        path = video_factory("clip.mkv", [1, 2, 3, 4])
        cache = FrameIndexCache(cache_dir)
        video = probe_video(path)
        FrameIndexer(cache, IndexParams()).get(video)

        token = CancellationToken()
        token.cancel()

        assert cache.load(video, IndexParams()) is not None


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestBuildIndexesCancellation:
    def test_queued_files_are_never_started(self, video_factory, cache_dir, tmp_path):
        rips = tmp_path / "rips"
        videos = [
            probe_video(video_factory(f"t{n}.mkv", [300 + n, 301 + n, 302 + n], directory=rips))
            for n in range(6)
        ]
        token = CancellationToken()
        token.cancel()
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(), token=token)

        with pytest.raises(OperationCancelledError):
            build_indexes(indexer, videos, max_workers=2)

        assert not list(Path(cache_dir).glob("*.npz"))

    def test_cancelling_mid_run_returns_promptly(self, video_factory, cache_dir, tmp_path):
        rips = tmp_path / "rips"
        videos = [
            probe_video(
                video_factory(
                    f"t{n}.mkv",
                    list(range(400 + n * 30, 430 + n * 30)),
                    seconds_per_frame=0.5,
                    directory=rips,
                )
            )
            for n in range(6)
        ]
        token = CancellationToken()
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=0.5), token=token)

        def cancel_soon() -> None:
            time.sleep(0.1)
            token.cancel()

        threading.Thread(target=cancel_soon).start()
        started = time.monotonic()
        with pytest.raises(OperationCancelledError):
            build_indexes(indexer, videos, max_workers=2)
        elapsed = time.monotonic() - started

        assert elapsed < 10.0, f"cancellation took {elapsed:.1f}s to unwind"
        assert token.tracked_process_count == 0
