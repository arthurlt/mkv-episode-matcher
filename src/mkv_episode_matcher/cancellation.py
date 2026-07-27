"""Cooperative cancellation for the worker threads.

A Python thread cannot be stopped from the outside, so Ctrl+C has to work by
consent. Three things have to happen together for it to feel instant:

* workers check a shared flag and stop asking for more work;
* queued-but-unstarted work is dropped rather than run to completion; and
* the child processes workers are blocked reading from are killed, because a
  thread parked in ``read()`` on an ``ffmpeg`` pipe will not look at any flag
  until that pipe produces something.

Missing the last point is what makes interruption feel broken: the flag is set
promptly, and then nothing happens for as long as the longest decode takes.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

__all__ = [
    "CancellationToken",
    "OperationCancelledError",
    "cancel_on_sigint",
]

logger = logging.getLogger(__name__)

#: How long a child process is given to exit after SIGTERM before SIGKILL.
_TERMINATE_GRACE_S = 3.0

#: Conventional shell exit code for "terminated by SIGINT".
EXIT_INTERRUPTED = 130


class OperationCancelledError(Exception):
    """Raised when work stopped early because cancellation was requested."""


class CancellationToken:
    """A thread-safe cancellation flag that also owns the child processes.

    Examples
    --------
    >>> token = CancellationToken()
    >>> token.cancelled
    False
    >>> token.cancel()
    >>> token.cancelled
    True
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()
        self._callbacks: list[Callable[[], None]] = []

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    @property
    def tracked_process_count(self) -> int:
        """Return how many child processes are currently being tracked."""
        with self._lock:
            return len(self._processes)

    def raise_if_cancelled(self) -> None:
        """Raise :class:`OperationCancelledError` if cancellation was requested."""
        if self._event.is_set():
            raise OperationCancelledError("cancelled")

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancelled, returning whether cancellation happened."""
        return self._event.wait(timeout)

    def cancel(self) -> None:
        """Request cancellation and kill every tracked child process.

        Safe to call from a signal handler and from any thread, and harmless to
        call more than once.
        """
        with self._lock:
            self._event.set()
            processes = list(self._processes)
            callbacks = list(self._callbacks)

        for process in processes:
            _terminate(process)
        for callback in callbacks:
            _run_callback(callback)

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register ``callback`` to run when cancellation is requested.

        If cancellation already happened, the callback runs immediately, so
        registering late can never mean never running.
        """
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(callback)
                return
        _run_callback(callback)

    @contextmanager
    def track(self, process: subprocess.Popen) -> Iterator[subprocess.Popen]:
        """Kill ``process`` if cancellation happens while the body runs.

        Raises
        ------
        OperationCancelledError
            If cancellation was already requested, in which case ``process`` is
            killed before the body is entered.
        """
        with self._lock:
            cancelled = self._event.is_set()
            if not cancelled:
                self._processes.add(process)

        if cancelled:
            _terminate(process)
            raise OperationCancelledError("cancelled before the process could be used")

        try:
            yield process
        finally:
            with self._lock:
                self._processes.discard(process)


def _terminate(process: subprocess.Popen) -> None:
    """Stop ``process``, escalating to SIGKILL if it ignores SIGTERM."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_TERMINATE_GRACE_S)
            return
        process.kill()
    except (OSError, ValueError) as error:
        logger.debug("could not terminate child process: %s", error)


def _run_callback(callback: Callable[[], None]) -> None:
    """Run a cancellation callback, never letting it break cancellation."""
    try:
        callback()
    except Exception:
        logger.exception("a cancellation callback failed")


@contextmanager
def cancel_on_sigint(
    token: CancellationToken, *, message: str = "\ninterrupted, stopping workers... "
) -> Iterator[CancellationToken]:
    """Make the first Ctrl+C cancel ``token`` and the second abort outright.

    The handler deliberately does not raise. Raising ``KeyboardInterrupt`` in
    the main thread would land inside whatever pool it was waiting on, and
    ``ThreadPoolExecutor.__exit__`` would then block until every in-flight task
    finished anyway. Setting the flag lets the workers unwind on their own
    terms while the main thread keeps waiting for them, which is both faster
    and tidier.

    Only the main thread may install signal handlers, so this is a no-op
    anywhere else rather than an error.
    """
    if threading.current_thread() is not threading.main_thread():
        yield token
        return

    previous = signal.getsignal(signal.SIGINT)

    def handle(_signum: int, _frame: object) -> None:
        # Restore the default so a second Ctrl+C is never swallowed, however
        # badly a worker is behaving.
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signal.SIGINT, previous)
        # os.write is a bare syscall; the logging module takes locks, and one
        # held by the interrupted thread would deadlock right here.
        with contextlib.suppress(OSError):
            os.write(2, f"{message}(press Ctrl+C again to abort)\n".encode())
        token.cancel()

    signal.signal(signal.SIGINT, handle)
    try:
        yield token
    finally:
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signal.SIGINT, previous)
