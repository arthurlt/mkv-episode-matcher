"""Parse and apply optional episode-number filters for disc-sized rips."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from .models import Episode

__all__ = [
    "claimed_episode_numbers",
    "filter_episodes_by_number",
    "parse_episode_code",
    "parse_episode_numbers",
]

_RANGE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_EPISODE_CODE = re.compile(r"[Ss](\d{1,2})[Ee](\d{1,3})")


def parse_episode_numbers(spec: str) -> frozenset[int]:
    """Parse a compact episode list such as ``1-7`` or ``1,3,5``.

    Parameters
    ----------
    spec
        Comma-separated numbers and inclusive ranges.

    Returns
    -------
    frozenset of int
        Episode numbers to include in matching.

    Raises
    ------
    ValueError
        If the string is empty or cannot be parsed.

    Examples
    --------
    >>> sorted(parse_episode_numbers("1-7"))
    [1, 2, 3, 4, 5, 6, 7]
    >>> sorted(parse_episode_numbers("2, 4-6"))
    [2, 4, 5, 6]
    >>> sorted(parse_episode_numbers("8-12"))
    [8, 9, 10, 11, 12]
    """
    text = spec.strip()
    if not text:
        raise ValueError("episode filter must not be empty")

    numbers: set[int] = set()
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        range_match = _RANGE.match(piece)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start > end:
                raise ValueError(f"invalid episode range: {piece}")
            numbers.update(range(start, end + 1))
            continue
        if not piece.isdigit():
            raise ValueError(f"invalid episode token: {piece!r}")
        numbers.add(int(piece))

    if not numbers:
        raise ValueError("episode filter must list at least one episode number")
    return frozenset(numbers)


def parse_episode_code(name: str) -> tuple[int, int] | None:
    """Return ``(season, episode)`` from a Plex-style file name, if present.

    Examples
    --------
    >>> parse_episode_code("Ted Lasso - S02E08 - Man City.mkv")
    (2, 8)
    >>> parse_episode_code("title_t00.mkv") is None
    True
    """
    match = _EPISODE_CODE.search(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def claimed_episode_numbers(paths: Iterable[Path], season: int) -> frozenset[int]:
    """Return episode numbers already identified in ``paths`` for ``season``.

    Files named ``…S02E08…`` are treated as claimed so a mixed folder of renamed
    Disc 1 rips and untitled Disc 2 titles can match the remaining episodes
    without competing against stills for episodes that are already done.
    """
    claimed: set[int] = set()
    for path in paths:
        parsed = parse_episode_code(path.name)
        if parsed is None:
            continue
        file_season, number = parsed
        if file_season == season:
            claimed.add(number)
    return frozenset(claimed)


def filter_episodes_by_number(episodes: list[Episode], numbers: frozenset[int]) -> list[Episode]:
    """Keep only episodes whose numbers appear in ``numbers``."""
    filtered = [episode for episode in episodes if episode.number in numbers]
    missing = numbers - {episode.number for episode in filtered}
    if missing:
        missing_list = ", ".join(f"E{n:02d}" for n in sorted(missing))
        raise ValueError(f"no metadata for episode(s) on this season: {missing_list}")
    return filtered
