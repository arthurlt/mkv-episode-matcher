"""Parse and apply optional episode-number filters for disc-sized rips."""

from __future__ import annotations

import re

from .models import Episode

__all__ = ["filter_episodes_by_number", "parse_episode_numbers"]

_RANGE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


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


def filter_episodes_by_number(episodes: list[Episode], numbers: frozenset[int]) -> list[Episode]:
    """Keep only episodes whose numbers appear in ``numbers``."""
    filtered = [episode for episode in episodes if episode.number in numbers]
    missing = numbers - {episode.number for episode in filtered}
    if missing:
        missing_list = ", ".join(f"E{n:02d}" for n in sorted(missing))
        raise ValueError(f"no metadata for episode(s) on this season: {missing_list}")
    return filtered
