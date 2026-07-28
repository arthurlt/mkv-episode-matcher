"""Global 1:1 assignment of files to episodes, and the final verdict rules.

Two independent guards stand between a visual hit and an automatic rename:

* the assignment is globally optimal and one-to-one, so a strong hit cannot be
  spent twice; and
* a file is only called ``matched`` when it and its episode are each other's
  clear best choice, by a configurable margin.

Anything short of that is reported as ``ambiguous`` with the evidence attached,
because a wrong rename costs a human more than an unanswered question does.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .models import (
    Episode,
    EpisodeScore,
    FileScores,
    MatchResult,
    MatchStatus,
    SkipReason,
    VideoFile,
)
from .score_visual import ScoringConfig

__all__ = ["assign", "confidence_of", "confidence_of_ncc", "solve_assignment"]

logger = logging.getLogger(__name__)

INF = float("inf")


def solve_assignment(cost_matrix: np.ndarray) -> list[int | None]:
    """Return the minimum-cost one-to-one assignment of rows to columns.

    Implements the Jonker-Volgenant form of the Hungarian algorithm. Entries of
    ``inf`` mark forbidden pairings and are replaced by a sentinel large enough
    that the optimum never uses one unless it has no alternative; such
    pairings come back as ``None``.

    Parameters
    ----------
    cost_matrix
        Two-dimensional array of costs, rows are files and columns episodes.
        Rectangular in either direction is fine.

    Returns
    -------
    list
        One entry per row: the column it was assigned, or ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> solve_assignment(np.array([[1.0, 9.0], [9.0, 1.0]]))
    [0, 1]
    >>> solve_assignment(np.array([[float("inf"), 2.0]]))
    [1]
    """
    matrix = np.asarray(cost_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    rows, columns = matrix.shape
    if rows == 0:
        return []
    if columns == 0:
        return [None] * rows

    forbidden = ~np.isfinite(matrix)
    finite = matrix[~forbidden]
    sentinel = (float(finite.max()) + 1.0) * (rows + columns) + 1.0 if finite.size else 1.0
    padded = np.where(forbidden, sentinel, matrix)

    if rows <= columns:
        assignment = _hungarian(padded)
    else:
        transposed = _hungarian(padded.T)
        assignment = [None] * rows
        for column, row in enumerate(transposed):
            if row is not None:
                assignment[row] = column

    return [
        None if column is None or forbidden[row][column] else column
        for row, column in enumerate(assignment)
    ]


def _hungarian(cost: np.ndarray) -> list[int | None]:
    """Solve a rectangular assignment problem with ``rows <= columns``."""
    rows, columns = cost.shape
    row_potential = [0.0] * (rows + 1)
    column_potential = [0.0] * (columns + 1)
    column_match = [0] * (columns + 1)
    path = [0] * (columns + 1)

    for row in range(1, rows + 1):
        column_match[0] = row
        current_column = 0
        minima = [INF] * (columns + 1)
        used = [False] * (columns + 1)

        while True:
            used[current_column] = True
            current_row = column_match[current_column]
            delta, next_column = INF, -1
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                candidate = (
                    cost[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if candidate < minima[column]:
                    minima[column] = candidate
                    path[column] = current_column
                if minima[column] < delta:
                    delta = minima[column]
                    next_column = column
            for column in range(columns + 1):
                if used[column]:
                    row_potential[column_match[column]] += delta
                    column_potential[column] -= delta
                else:
                    minima[column] -= delta
            current_column = next_column
            if column_match[current_column] == 0:
                break

        while current_column:
            previous = path[current_column]
            column_match[current_column] = column_match[previous]
            current_column = previous

    assignment: list[int | None] = [None] * rows
    for column in range(1, columns + 1):
        if column_match[column]:
            assignment[column_match[column] - 1] = column - 1
    return assignment


def assign(
    scored: Sequence[FileScores],
    *,
    config: ScoringConfig,
    skipped: dict[Path, SkipReason] | None = None,
    unindexed: Sequence[VideoFile] = (),
    closed_world: bool = False,
) -> list[MatchResult]:
    """Turn per-file episode scores into final, mutually consistent verdicts.

    Parameters
    ----------
    scored
        Per-file episode scores from :mod:`mkv_episode_matcher.score_visual`.
    config
        Thresholds; ``match_threshold`` decides feasibility and
        ``decision_gap`` decides confidence.
    skipped
        Files excluded before matching, with the reason.
    unindexed
        Files that could not be decoded, reported as unmatched.
    closed_world
        When true (``--episodes`` filter), soft-threshold NCC hits may be
        accepted if the mutual-best soft gap holds. Full-season runs keep the
        hard threshold so weak cross-episode collisions stay unmatched.

    Returns
    -------
    list of MatchResult
        One result per file, sorted by file name.
    """
    results: list[MatchResult] = []

    if scored:
        results.extend(_assign_scored(scored, config, closed_world=closed_world))

    for path, reason in (skipped or {}).items():
        results.append(
            MatchResult(
                video=VideoFile(path, duration_s=0.0, size_bytes=0, mtime_ns=0),
                status=MatchStatus.SKIPPED,
                skip_reason=reason,
                notes=[f"skipped: {reason.value}"],
            )
        )

    for video in unindexed:
        results.append(
            MatchResult(
                video=video,
                status=MatchStatus.UNMATCHED,
                notes=["no frame index could be built for this file"],
            )
        )

    return sorted(results, key=lambda result: result.video.name)


def _assign_scored(
    scored: Sequence[FileScores],
    config: ScoringConfig,
    *,
    closed_world: bool = False,
) -> list[MatchResult]:
    """Solve the assignment and apply the mutual-best decision rules."""
    episodes = _episode_axis(scored)
    uses_ncc = _uses_ncc(scored)
    cost_matrix = np.array(
        [
            [
                score.cost if _feasible(score, config, closed_world=closed_world) else INF
                for score in _ordered(file_scores, episodes)
            ]
            for file_scores in scored
        ],
        dtype=float,
    )

    reject_cost = None
    if uses_ncc:
        # Extra column lets a file stay unmatched rather than absorb a weak
        # pairing by elimination. Cost equals the feasibility boundary.
        floor = (
            config.verify_soft_threshold
            if closed_world
            else config.verify_threshold
        )
        reject_cost = 1.0 - floor
        cost_matrix = np.concatenate(
            [
                cost_matrix,
                np.full((cost_matrix.shape[0], 1), reject_cost, dtype=float),
            ],
            axis=1,
        )

    assignment = solve_assignment(cost_matrix)
    episode_columns = len(episodes)

    results = []
    for row, file_scores in enumerate(scored):
        column = assignment[row]
        if reject_cost is not None and column is not None and column >= episode_columns:
            column = None
        results.append(
            _verdict(
                file_scores,
                cost_matrix[:, :episode_columns],
                row,
                column,
                episodes,
                config,
                uses_ncc=uses_ncc,
                closed_world=closed_world,
            )
        )
    return results


def _uses_ncc(scored: Sequence[FileScores]) -> bool:
    """Return whether verification has attached NCC scores to use for decisions."""
    return any(score.ncc is not None for file_scores in scored for score in file_scores.scores)


def _episode_axis(scored: Sequence[FileScores]) -> list[Episode]:
    """Return every episode any file was scored against, ordered by number."""
    by_number: dict[int, Episode] = {}
    for file_scores in scored:
        for score in file_scores.scores:
            by_number.setdefault(score.episode.number, score.episode)
    return [by_number[number] for number in sorted(by_number)]


def _ordered(file_scores: FileScores, episodes: Sequence[Episode]) -> list[EpisodeScore]:
    """Return this file's scores in the shared episode order.

    An episode this file was never scored against contributes an infinite cost
    rather than an error, so a partially scored input degrades instead of
    aborting.
    """
    by_number = {score.episode.number: score for score in file_scores.scores}
    return [
        by_number.get(
            episode.number,
            EpisodeScore(episode=episode, best_hit=None, supporting_stills=0, cost=INF),
        )
        for episode in episodes
    ]


def _feasible(
    score: EpisodeScore, config: ScoringConfig, *, closed_world: bool = False
) -> bool:
    """Return whether a score is a usable hit at all.

    With NCC verification, feasibility is ``ncc >= verify_threshold``, or
    ``ncc >= verify_soft_threshold`` when matching a closed episode set.
    Otherwise it is judged on the pHash cost, so the dHash collision penalty
    can push a suspicious hit out of contention entirely.
    """
    if score.ncc is not None:
        floor = (
            config.verify_soft_threshold
            if closed_world
            else config.verify_threshold
        )
        return score.ncc >= floor
    return score.best_hit is not None and score.cost <= config.match_threshold


def _decision_gap(
    config: ScoringConfig,
    *,
    uses_ncc: bool,
    soft: bool = False,
) -> float:
    """Return the mutual-best margin for the active cost regime."""
    if not uses_ncc:
        return config.decision_gap
    return config.verify_soft_gap if soft else config.verify_gap


def _verdict(
    file_scores: FileScores,
    cost_matrix: np.ndarray,
    row: int,
    column: int | None,
    episodes: Sequence[Episode],
    config: ScoringConfig,
    *,
    uses_ncc: bool = False,
    closed_world: bool = False,
) -> MatchResult:
    """Decide one file's status from the assignment and the surrounding costs."""
    video = file_scores.video
    ordered = _ordered(file_scores, episodes)
    ranked = sorted(ordered, key=lambda score: (score.cost, score.episode.number))
    best = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    best_hit = best.best_hit if best else None
    soft = bool(
        uses_ncc
        and closed_world
        and best is not None
        and best.ncc is not None
        and best.ncc < config.verify_threshold
    )
    gap_needed = _decision_gap(config, uses_ncc=uses_ncc, soft=soft)

    if best is None or best_hit is None or not _feasible(best, config, closed_world=closed_world):
        note = (
            "no still verified above the NCC threshold"
            if uses_ncc
            else "no still matched this file below the threshold"
        )
        return MatchResult(
            video=video,
            status=MatchStatus.UNMATCHED,
            runner_up=best.episode if best and best.best_hit else None,
            runner_up_cost=None if best is None or best.cost == INF else best.cost,
            ncc=best.ncc if best else None,
            notes=[note],
        )

    file_gap = INF if runner_up is None else runner_up.cost - best.cost
    episode_gap = _episode_side_gap(cost_matrix, row, ranked[0], episodes)
    chosen_by_solver = column is not None and episodes[column].number == best.episode.number
    if uses_ncc and best.ncc is not None:
        confidence = confidence_of_ncc(best.ncc, min(file_gap, episode_gap), config)
    else:
        confidence = confidence_of(best.cost, min(file_gap, episode_gap), config)

    if chosen_by_solver and file_gap >= gap_needed and episode_gap >= gap_needed:
        if uses_ncc and best.ncc is not None:
            logger.info(
                "matched %s -> %s (ncc %.3f at %.1fs, distance %d, %d supporting stills)",
                video.name,
                best.episode.code,
                best.ncc,
                best_hit.timestamp,
                best_hit.distance,
                best.supporting_stills,
            )
        else:
            logger.info(
                "matched %s -> %s (distance %d at %.1fs, %d supporting stills)",
                video.name,
                best.episode.code,
                best_hit.distance,
                best_hit.timestamp,
                best.supporting_stills,
            )
        notes = (
            [
                f"accepted under soft NCC threshold "
                f"({best.ncc:.3f} < {config.verify_threshold:.2f}; "
                f"closed episode set)"
            ]
            if soft and best.ncc is not None
            else []
        )
        return MatchResult(
            video=video,
            status=MatchStatus.MATCHED,
            episode=best.episode,
            hit=best_hit,
            cost=best.cost,
            runner_up=runner_up.episode if runner_up else None,
            runner_up_cost=None if runner_up is None or runner_up.cost == INF else runner_up.cost,
            supporting_stills=best.supporting_stills,
            confidence=confidence,
            ncc=best.ncc,
            notes=notes,
        )

    notes = []
    if not chosen_by_solver:
        notes.append(
            f"another file is a better fit for {best.episode.code}; "
            "the global assignment moved this one"
        )
    if file_gap < gap_needed and runner_up is not None:
        notes.append(
            f"{best.episode.code} and {runner_up.episode.code} are within "
            f"{file_gap:.3f} of each other"
        )
    if episode_gap < gap_needed:
        notes.append(f"another file matches {best.episode.code} almost as well")

    logger.info("ambiguous %s: %s", video.name, "; ".join(notes))
    return MatchResult(
        video=video,
        status=MatchStatus.AMBIGUOUS,
        episode=None,
        hit=best_hit,
        cost=best.cost,
        runner_up=runner_up.episode if runner_up else best.episode,
        runner_up_cost=None if runner_up is None or runner_up.cost == INF else runner_up.cost,
        supporting_stills=best.supporting_stills,
        confidence=confidence,
        ncc=best.ncc,
        notes=notes or [f"best candidate is {best.episode.code}"],
    )


def confidence_of(cost: float, gap: float, config: ScoringConfig) -> float:
    """Rate a pHash candidate from 0 to 1 on its two independent weaknesses.

    This is an ordering aid for triage, not a probability. It is the weaker of
    two margins: how far below the accept threshold the hit landed, and how
    decisively it beat the runner-up. A result is only as trustworthy as its
    weakest margin, so the minimum is taken rather than an average.

    Examples
    --------
    >>> config = ScoringConfig(match_threshold=12, decision_gap=4.0, verify=False)
    >>> confidence_of(0.0, 20.0, config)
    1.0
    >>> confidence_of(12.0, 20.0, config)
    0.0
    >>> round(confidence_of(6.0, 2.0, config), 2)
    0.5
    """
    if config.match_threshold <= 0:
        distance_margin = 1.0 if cost <= 0 else 0.0
    else:
        distance_margin = 1.0 - cost / config.match_threshold
    gap_margin = 1.0 if config.decision_gap <= 0 else gap / config.decision_gap
    return round(max(0.0, min(1.0, distance_margin, gap_margin)), 3)


def confidence_of_ncc(ncc_value: float, gap: float, config: ScoringConfig) -> float:
    """Rate an NCC-verified candidate from 0 to 1.

    Examples
    --------
    >>> config = ScoringConfig(verify_threshold=0.65, verify_gap=0.05)
    >>> confidence_of_ncc(1.0, 0.2, config)
    1.0
    >>> confidence_of_ncc(0.65, 0.2, config)
    0.0
    """
    span = 1.0 - config.verify_threshold
    if span <= 0:
        ncc_margin = 1.0 if ncc_value >= config.verify_threshold else 0.0
    else:
        ncc_margin = (ncc_value - config.verify_threshold) / span
    gap_margin = 1.0 if config.verify_gap <= 0 else gap / config.verify_gap
    return round(max(0.0, min(1.0, ncc_margin, gap_margin)), 3)


def _episode_side_gap(
    cost_matrix: np.ndarray, row: int, best: EpisodeScore, episodes: Sequence[Episode]
) -> float:
    """Return how much worse the next-best *file* is for this file's best episode."""
    column = next(
        (index for index, episode in enumerate(episodes) if episode.number == best.episode.number),
        None,
    )
    if column is None:
        return INF
    competitors = np.delete(cost_matrix[:, column], row)
    if competitors.size == 0:
        return INF
    return float(competitors.min()) - cost_matrix[row][column]
