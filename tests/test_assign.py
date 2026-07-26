"""Tests for the 1:1 assignment solver and the final match/ambiguous rules."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mkv_episode_matcher.assign import assign, confidence_of, solve_assignment
from mkv_episode_matcher.models import (
    Episode,
    EpisodeScore,
    FileScores,
    MatchStatus,
    SkipReason,
    Still,
    StillHit,
    VideoFile,
)
from mkv_episode_matcher.score_visual import ScoringConfig

INF = float("inf")


def episode(number: int) -> Episode:
    return Episode(season=2, number=number, title=f"Episode {number}")


def video(name: str) -> VideoFile:
    return VideoFile(Path(name), duration_s=1320.0, size_bytes=1, mtime_ns=1)


def scores_for(name: str, costs: dict[int, float], supporting: int = 1) -> FileScores:
    entries = []
    for number, cost in costs.items():
        hit = (
            None
            if cost == INF
            else StillHit(
                still=Still("tmdb", f"https://img/{number}.jpg"),
                distance=int(cost),
                timestamp=float(number) * 60.0,
                dhash_distance=2,
            )
        )
        entries.append(
            EpisodeScore(
                episode=episode(number),
                best_hit=hit,
                supporting_stills=supporting if hit else 0,
                cost=cost,
            )
        )
    return FileScores(video=video(name), scores=tuple(entries))


def brute_force_cost(matrix: np.ndarray) -> float:
    rows, columns = matrix.shape
    best = INF
    for combination in itertools.permutations(range(columns), rows):
        total = sum(matrix[row][column] for row, column in enumerate(combination))
        best = min(best, total)
    return best


class TestSolveAssignment:
    def test_picks_the_obvious_diagonal(self):
        matrix = np.array([[1.0, 9.0], [9.0, 1.0]])

        assert solve_assignment(matrix) == [0, 1]

    def test_prefers_the_globally_cheaper_pairing_over_the_greedy_one(self):
        matrix = np.array([[1.0, 2.0], [1.5, 30.0]])

        assert solve_assignment(matrix) == [1, 0]

    def test_handles_more_episodes_than_files(self):
        matrix = np.array([[5.0, 1.0, 9.0]])

        assert solve_assignment(matrix) == [1]

    def test_handles_more_files_than_episodes(self):
        matrix = np.array([[1.0], [4.0], [9.0]])

        result = solve_assignment(matrix)

        assert result.count(0) == 1
        assert result.count(None) == 2

    def test_forbidden_pairs_are_left_unassigned(self):
        matrix = np.array([[INF, 3.0], [INF, INF]])

        assert solve_assignment(matrix) == [1, None]

    def test_an_all_forbidden_matrix_assigns_nothing(self):
        matrix = np.full((2, 2), INF)

        assert solve_assignment(matrix) == [None, None]

    def test_empty_input_is_handled(self):
        assert solve_assignment(np.zeros((0, 3))) == []
        assert solve_assignment(np.zeros((2, 0))) == [None, None]

    @settings(max_examples=60, deadline=None)
    @given(
        st.integers(min_value=1, max_value=4).flatmap(
            lambda rows: st.integers(min_value=rows, max_value=5).flatmap(
                lambda columns: st.lists(
                    st.lists(
                        st.floats(min_value=0, max_value=50, allow_nan=False),
                        min_size=columns,
                        max_size=columns,
                    ),
                    min_size=rows,
                    max_size=rows,
                )
            )
        )
    )
    def test_matches_brute_force_optimum(self, values):
        matrix = np.array(values, dtype=float)

        result = solve_assignment(matrix)

        total = sum(matrix[row][column] for row, column in enumerate(result) if column is not None)
        assert total == pytest.approx(brute_force_cost(matrix), abs=1e-6)

    @settings(max_examples=60, deadline=None)
    @given(
        st.lists(
            st.lists(
                st.one_of(st.floats(min_value=0, max_value=30, allow_nan=False), st.just(INF)),
                min_size=1,
                max_size=4,
            ),
            min_size=1,
            max_size=4,
        )
    )
    def test_assignment_is_always_one_to_one(self, values):
        width = max(len(row) for row in values)
        matrix = np.array([row + [INF] * (width - len(row)) for row in values], dtype=float)

        result = solve_assignment(matrix)

        assigned = [column for column in result if column is not None]
        assert len(assigned) == len(set(assigned))
        assert len(result) == matrix.shape[0]

    @settings(max_examples=40, deadline=None)
    @given(
        st.lists(
            st.lists(st.floats(min_value=0, max_value=20, allow_nan=False), min_size=3, max_size=3),
            min_size=3,
            max_size=3,
        )
    )
    def test_never_assigns_a_forbidden_pair(self, values):
        matrix = np.array(values, dtype=float)
        matrix[0][0] = INF

        result = solve_assignment(matrix)

        assert result[0] != 0


class TestConfidence:
    config = ScoringConfig(match_threshold=12, decision_gap=4.0)

    def test_an_exact_hit_with_a_wide_gap_is_fully_confident(self):
        assert confidence_of(0.0, 30.0, self.config) == 1.0

    def test_a_hit_at_the_threshold_has_no_confidence(self):
        assert confidence_of(12.0, 30.0, self.config) == 0.0

    def test_a_dead_heat_has_no_confidence_however_close_the_hit(self):
        assert confidence_of(0.0, 0.0, self.config) == 0.0

    def test_the_weaker_of_the_two_margins_wins(self):
        strong_distance_weak_gap = confidence_of(1.0, 1.0, self.config)
        weak_distance_strong_gap = confidence_of(11.0, 30.0, self.config)

        assert strong_distance_weak_gap == pytest.approx(0.25)
        assert weak_distance_strong_gap == pytest.approx(1 / 12, abs=0.01)

    def test_an_infinite_gap_is_bounded_by_the_distance_margin(self):
        assert confidence_of(6.0, INF, self.config) == pytest.approx(0.5)

    @settings(max_examples=50, deadline=None)
    @given(
        st.floats(min_value=0, max_value=64, allow_nan=False),
        st.floats(min_value=0, max_value=100, allow_nan=False),
    )
    def test_confidence_always_lands_between_zero_and_one(self, cost, gap):
        assert 0.0 <= confidence_of(cost, gap, self.config) <= 1.0

    def test_a_zero_gap_requirement_never_penalises_the_gap(self):
        forgiving = ScoringConfig(match_threshold=12, decision_gap=0.0)

        assert confidence_of(0.0, 0.0, forgiving) == 1.0


class TestAssign:
    config = ScoringConfig(match_threshold=12, decision_gap=4.0)

    def test_a_clean_match_is_reported_as_fully_confident(self):
        scored = [scores_for("a.mkv", {1: 0.0, 2: 40.0})]

        assert assign(scored, config=self.config)[0].confidence == 1.0

    def test_an_ambiguous_result_carries_a_low_confidence(self):
        scored = [scores_for("a.mkv", {1: 5.0, 2: 6.0})]

        result = assign(scored, config=self.config)[0]

        assert result.status is MatchStatus.AMBIGUOUS
        assert result.confidence is not None
        assert result.confidence < 0.5

    def test_an_unmatched_result_has_no_confidence(self):
        scored = [scores_for("a.mkv", {1: INF})]

        assert assign(scored, config=self.config)[0].confidence is None

    def test_clean_diagonal_matches_every_file(self):
        scored = [
            scores_for("a.mkv", {1: 1.0, 2: 30.0, 3: 40.0}),
            scores_for("b.mkv", {1: 33.0, 2: 2.0, 3: 41.0}),
            scores_for("c.mkv", {1: 35.0, 2: 36.0, 3: 3.0}),
        ]

        results = assign(scored, config=self.config)

        assert [result.status for result in results] == [MatchStatus.MATCHED] * 3
        assert [result.episode.number for result in results] == [1, 2, 3]

    def test_disc_order_does_not_influence_the_outcome(self):
        forward = [
            scores_for("t00.mkv", {1: 30.0, 2: 30.0, 3: 2.0}),
            scores_for("t01.mkv", {1: 1.0, 2: 31.0, 3: 32.0}),
            scores_for("t02.mkv", {1: 33.0, 2: 3.0, 3: 34.0}),
        ]
        reversed_order = list(reversed(forward))

        forward_map = {r.video.name: r.episode.number for r in assign(forward, config=self.config)}
        reverse_map = {
            r.video.name: r.episode.number for r in assign(reversed_order, config=self.config)
        }

        assert forward_map == reverse_map == {"t00.mkv": 3, "t01.mkv": 1, "t02.mkv": 2}

    def test_a_file_with_no_hit_is_unmatched(self):
        scored = [scores_for("a.mkv", {1: INF, 2: INF})]

        results = assign(scored, config=self.config)

        assert results[0].status is MatchStatus.UNMATCHED
        assert results[0].episode is None

    def test_a_hit_above_the_threshold_is_unmatched(self):
        scored = [scores_for("a.mkv", {1: 20.0, 2: 40.0})]

        results = assign(scored, config=self.config)

        assert results[0].status is MatchStatus.UNMATCHED

    def test_a_narrow_gap_is_ambiguous_not_a_coin_flip(self):
        scored = [scores_for("a.mkv", {1: 5.0, 2: 6.0, 3: 40.0})]

        results = assign(scored, config=self.config)

        assert results[0].status is MatchStatus.AMBIGUOUS
        assert results[0].runner_up.number == 2

    def test_two_files_contending_for_one_episode_are_ambiguous(self):
        scored = [
            scores_for("a.mkv", {1: 2.0, 2: 40.0}),
            scores_for("b.mkv", {1: 3.0, 2: 41.0}),
        ]

        results = assign(scored, config=self.config)

        assert {result.status for result in results} == {MatchStatus.AMBIGUOUS}

    def test_an_ambiguous_result_still_reports_its_best_candidate(self):
        scored = [scores_for("a.mkv", {1: 5.0, 2: 6.0})]

        results = assign(scored, config=self.config)

        assert results[0].episode is None
        assert results[0].notes

    def test_matched_results_carry_the_still_and_timestamp(self):
        scored = [scores_for("a.mkv", {1: 1.0, 2: 40.0})]

        result = assign(scored, config=self.config)[0]

        assert result.hit.timestamp == pytest.approx(60.0)
        assert result.hit.still.url.endswith("1.jpg")

    def test_matched_results_report_the_runner_up(self):
        result = assign([scores_for("a.mkv", {1: 1.0, 2: 40.0})], config=self.config)[0]

        assert result.runner_up.number == 2
        assert result.gap == pytest.approx(39.0)

    def test_skipped_files_are_reported_without_being_matched(self):
        skipped = {Path("menu.mkv"): SkipReason.TOO_SHORT}

        results = assign([], config=self.config, skipped=skipped)

        assert results[0].status is MatchStatus.SKIPPED
        assert results[0].skip_reason is SkipReason.TOO_SHORT
        assert results[0].renameable is False

    def test_unindexed_files_are_reported_as_unmatched(self):
        results = assign([], config=self.config, unindexed=[video("broken.mkv")])

        assert results[0].status is MatchStatus.UNMATCHED
        assert "index" in " ".join(results[0].notes).lower()

    def test_no_episode_is_ever_assigned_twice(self):
        scored = [
            scores_for("a.mkv", {1: 1.0, 2: 2.0, 3: 3.0}),
            scores_for("b.mkv", {1: 1.5, 2: 2.5, 3: 3.5}),
            scores_for("c.mkv", {1: 2.0, 2: 3.0, 3: 4.0}),
        ]

        results = assign(scored, config=self.config)

        matched = [r.episode.number for r in results if r.status is MatchStatus.MATCHED]
        assert len(matched) == len(set(matched))

    def test_more_files_than_episodes_leaves_the_extras_unmatched(self):
        scored = [
            scores_for("a.mkv", {1: 1.0}),
            scores_for("b.mkv", {1: 30.0}),
        ]

        results = assign(scored, config=self.config)

        by_name = {result.video.name: result for result in results}
        assert by_name["a.mkv"].status is MatchStatus.MATCHED
        assert by_name["b.mkv"].status is MatchStatus.UNMATCHED

    def test_results_are_sorted_by_file_name(self):
        scored = [scores_for("c.mkv", {1: 1.0}), scores_for("a.mkv", {2: 1.0})]

        results = assign(scored, config=self.config)

        assert [result.video.name for result in results] == ["a.mkv", "c.mkv"]

    def test_multi_still_agreement_breaks_a_tie(self):
        """Same raw distance, but one episode has several stills agreeing."""
        weak = scores_for("a.mkv", {1: 8.0})
        strong_scores = [
            *weak.scores,
            EpisodeScore(
                episode=episode(2),
                best_hit=StillHit(Still("tmdb", "https://img/2.jpg"), 8, 120.0, 2),
                supporting_stills=4,
                cost=8.0 - 2.25,
            ),
        ]
        scored = [FileScores(video=video("a.mkv"), scores=tuple(strong_scores))]

        result = assign(scored, config=ScoringConfig(match_threshold=12, decision_gap=2.0))[0]

        assert result.status is MatchStatus.MATCHED
        assert result.episode.number == 2
