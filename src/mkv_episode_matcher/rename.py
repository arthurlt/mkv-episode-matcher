"""Plex-style renaming, with every safety rail switched on by default.

Renaming is the only destructive thing this tool does, so it is gated three
ways: only ``matched`` results are ever planned, an existing target is never
overwritten without ``--force``, and nothing happens at all unless the caller
explicitly asks to leave dry-run mode.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .models import Episode, MatchResult

__all__ = [
    "RenameOutcome",
    "RenamePlan",
    "apply_renames",
    "plan_renames",
    "plex_filename",
    "sanitize_component",
]

logger = logging.getLogger(__name__)

_REPLACEMENTS = {
    "/": "-",
    "\\": "-",
    ":": " -",
    "*": "",
    "?": "",
    '"': "'",
    "<": "(",
    ">": ")",
    "|": "-",
}
_WHITESPACE = re.compile(r"\s+")


def sanitize_component(text: str) -> str:
    """Make ``text`` safe to use as a file name component.

    Characters that are illegal on Windows or awkward in shells are replaced
    with readable equivalents rather than stripped, so titles stay legible.

    Examples
    --------
    >>> sanitize_component("Who/What: Why?")
    'Who-What - Why'
    >>> sanitize_component("///")
    'untitled'
    """
    for character, replacement in _REPLACEMENTS.items():
        text = text.replace(character, replacement)
    text = _WHITESPACE.sub(" ", text).strip(" .-")
    return text or "untitled"


def plex_filename(series: str, episode: Episode, suffix: str) -> str:
    """Return the Plex-style file name for ``episode``.

    Examples
    --------
    >>> from mkv_episode_matcher.models import Episode
    >>> plex_filename("Test Precinct", Episode(2, 1, "Chipped Beef"), ".mkv")
    'Test Precinct - S02E01 - Chipped Beef.mkv'
    >>> plex_filename("Test Precinct", Episode(2, 7, ""), ".mkv")
    'Test Precinct - S02E07.mkv'
    """
    parts = [sanitize_component(series), episode.code]
    if episode.title.strip():
        parts.append(sanitize_component(episode.title))
    return " - ".join(parts) + suffix


@dataclass(frozen=True, slots=True)
class RenamePlan:
    """A single proposed rename."""

    source: Path
    target: Path
    episode: Episode


@dataclass(frozen=True, slots=True)
class RenameOutcome:
    """What actually happened to a :class:`RenamePlan`."""

    plan: RenamePlan
    applied: bool
    reason: str

    @property
    def description(self) -> str:
        """Return a one-line description of the outcome.

        Examples
        --------
        >>> from mkv_episode_matcher.models import Episode
        >>> plan = RenamePlan(Path("a.mkv"), Path("b.mkv"), Episode(1, 1, "X"))
        >>> RenameOutcome(plan, False, "dry run").description
        'would rename a.mkv -> b.mkv (dry run)'
        """
        verb = "renamed" if self.applied else "would rename"
        return f"{verb} {self.plan.source.name} -> {self.plan.target.name} ({self.reason})"


def plan_renames(results: Sequence[MatchResult], *, series: str) -> list[RenamePlan]:
    """Return renames for the confidently matched results only.

    Ambiguous, unmatched, and skipped results are deliberately excluded: there
    is no duration-only or best-guess path to a rename.
    """
    plans = []
    for result in results:
        if not result.renameable or result.episode is None:
            continue
        source = result.video.path
        target = source.with_name(plex_filename(series, result.episode, source.suffix))
        if target == source:
            logger.debug("%s is already correctly named", source.name)
            continue
        plans.append(RenamePlan(source=source, target=target, episode=result.episode))
    return plans


def apply_renames(
    plans: Sequence[RenamePlan], *, dry_run: bool = True, force: bool = False
) -> list[RenameOutcome]:
    """Carry out ``plans``, refusing to clobber anything unless ``force``.

    Parameters
    ----------
    plans
        Renames produced by :func:`plan_renames`.
    dry_run
        When true (the default) nothing is touched; the outcomes describe what
        would have happened.
    force
        Allow replacing an existing file at the target path.

    Returns
    -------
    list of RenameOutcome
        One outcome per plan, in order.
    """
    outcomes: list[RenameOutcome] = []
    claimed: set[Path] = set()

    for plan in plans:
        reason = _blocker(plan, claimed, force=force)
        if reason is not None:
            logger.warning("not renaming %s: %s", plan.source.name, reason)
            outcomes.append(RenameOutcome(plan, applied=False, reason=reason))
            continue
        if dry_run:
            claimed.add(plan.target)
            outcomes.append(RenameOutcome(plan, applied=False, reason="dry run"))
            continue
        try:
            plan.source.replace(plan.target)
        except OSError as error:
            logger.error("failed to rename %s: %s", plan.source.name, error)
            outcomes.append(RenameOutcome(plan, applied=False, reason=str(error)))
            continue
        claimed.add(plan.target)
        logger.info("renamed %s -> %s", plan.source.name, plan.target.name)
        outcomes.append(RenameOutcome(plan, applied=True, reason="renamed"))

    return outcomes


def _blocker(plan: RenamePlan, claimed: set[Path], *, force: bool) -> str | None:
    """Return why ``plan`` must not run, or ``None`` if it is safe."""
    if not plan.source.exists():
        return "source file no longer exists"
    if plan.target in claimed:
        return "another file already claimed this name in this run"
    if plan.target.exists() and not force:
        return "target already exists; re-run with --force to replace it"
    return None
