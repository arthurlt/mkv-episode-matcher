"""Logging setup for the command line."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["configure_logging"]

_LEVELS = [logging.WARNING, logging.INFO, logging.DEBUG]


def configure_logging(verbosity: int = 0, *, quiet: bool = False, console: Console | None = None) -> int:
    """Install a Rich log handler and return the level that was selected.

    Parameters
    ----------
    verbosity
        ``0`` warns only, ``1`` (``-v``) narrates the run, ``2`` (``-vv``)
        includes every ffmpeg invocation and cache decision.
    quiet
        Suppress everything below an error.
    console
        Console to log to; defaults to stderr.

    Returns
    -------
    int
        The configured logging level.

    Examples
    --------
    >>> import logging
    >>> configure_logging(1) == logging.INFO
    True
    """
    level = logging.ERROR if quiet else _LEVELS[min(verbosity, len(_LEVELS) - 1)]
    handler = RichHandler(
        console=console or Console(stderr=True),
        rich_tracebacks=True,
        show_path=level <= logging.DEBUG,
        show_time=level <= logging.DEBUG,
        markup=False,
    )
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[handler], force=True
    )
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))
    return level
