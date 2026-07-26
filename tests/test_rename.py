"""Tests for Plex-style rename planning and safe application."""

from __future__ import annotations

from pathlib import Path

import pytest

from mkv_episode_matcher.models import (
    Episode,
    MatchResult,
    MatchStatus,
    SkipReason,
    Still,
    StillHit,
    VideoFile,
)
from mkv_episode_matcher.rename import (
    RenameOutcome,
    apply_renames,
    plan_renames,
    plex_filename,
    sanitize_component,
)


def result_for(path: Path, number: int, title: str, status=MatchStatus.MATCHED) -> MatchResult:
    stat = path.stat()
    return MatchResult(
        video=VideoFile(path, 1320.0, stat.st_size, stat.st_mtime_ns),
        status=status,
        episode=Episode(season=2, number=number, title=title),
        hit=StillHit(Still("tmdb", "https://img/a.jpg"), 2, 100.0, 2),
        cost=2.0,
    )


class TestSanitize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Chipped Beef", "Chipped Beef"),
            ("Who/What", "Who-What"),
            ("A: B", "A - B"),
            ('Say "Uncle"', "Say 'Uncle'"),
            ("Trailing.", "Trailing"),
            ("  padded  ", "padded"),
            ("What?", "What"),
        ],
    )
    def test_produces_a_filesystem_safe_component(self, raw, expected):
        assert sanitize_component(raw) == expected

    def test_collapses_runs_of_whitespace(self):
        assert sanitize_component("too    many   spaces") == "too many spaces"

    def test_never_returns_an_empty_string(self):
        assert sanitize_component("///") == "untitled"


class TestPlexFilename:
    def test_uses_the_standard_layout(self):
        episode = Episode(season=2, number=1, title="Chipped Beef")

        assert plex_filename("Test Precinct", episode, ".mkv") == (
            "Test Precinct - S02E01 - Chipped Beef.mkv"
        )

    def test_omits_an_empty_title(self):
        episode = Episode(season=2, number=7, title="")

        assert plex_filename("Test Precinct", episode, ".mkv") == "Test Precinct - S02E07.mkv"

    def test_pads_double_digit_seasons_and_episodes(self):
        episode = Episode(season=11, number=23, title="X")

        assert "S11E23" in plex_filename("Show", episode, ".mkv")

    def test_preserves_the_original_extension(self):
        episode = Episode(season=1, number=1, title="X")

        assert plex_filename("Show", episode, ".MKV").endswith(".MKV")


class TestPlanRenames:
    def test_plans_a_rename_for_a_matched_file(self, tmp_path):
        source = tmp_path / "title_t00.mkv"
        source.touch()

        plans = plan_renames([result_for(source, 1, "Chipped Beef")], series="Test Precinct")

        assert len(plans) == 1
        assert plans[0].target.name == "Test Precinct - S02E01 - Chipped Beef.mkv"
        assert plans[0].target.parent == tmp_path

    @pytest.mark.parametrize(
        "status", [MatchStatus.AMBIGUOUS, MatchStatus.UNMATCHED, MatchStatus.SKIPPED]
    )
    def test_never_plans_a_rename_for_an_unconfident_result(self, tmp_path, status):
        source = tmp_path / "title_t00.mkv"
        source.touch()

        assert plan_renames([result_for(source, 1, "X", status)], series="Show") == []

    def test_skips_a_file_that_is_already_correctly_named(self, tmp_path):
        source = tmp_path / "Show - S02E01 - X.mkv"
        source.touch()

        assert plan_renames([result_for(source, 1, "X")], series="Show") == []


class TestApplyRenames:
    def test_dry_run_leaves_the_filesystem_alone(self, tmp_path):
        source = tmp_path / "title_t00.mkv"
        source.write_bytes(b"data")
        plans = plan_renames([result_for(source, 1, "X")], series="Show")

        outcomes = apply_renames(plans, dry_run=True)

        assert source.exists()
        assert outcomes[0].applied is False
        assert outcomes[0].reason == "dry run"

    def test_renames_the_file_when_applied(self, tmp_path):
        source = tmp_path / "title_t00.mkv"
        source.write_bytes(b"data")
        plans = plan_renames([result_for(source, 1, "X")], series="Show")

        outcomes = apply_renames(plans, dry_run=False)

        assert not source.exists()
        assert (tmp_path / "Show - S02E01 - X.mkv").read_bytes() == b"data"
        assert outcomes[0].applied is True

    def test_refuses_to_overwrite_an_existing_file(self, tmp_path):
        source = tmp_path / "title_t00.mkv"
        source.write_bytes(b"new")
        existing = tmp_path / "Show - S02E01 - X.mkv"
        existing.write_bytes(b"old")
        plans = plan_renames([result_for(source, 1, "X")], series="Show")

        outcomes = apply_renames(plans, dry_run=False)

        assert existing.read_bytes() == b"old"
        assert source.exists()
        assert outcomes[0].applied is False
        assert "exists" in outcomes[0].reason

    def test_force_replaces_an_existing_file(self, tmp_path):
        source = tmp_path / "title_t00.mkv"
        source.write_bytes(b"new")
        existing = tmp_path / "Show - S02E01 - X.mkv"
        existing.write_bytes(b"old")
        plans = plan_renames([result_for(source, 1, "X")], series="Show")

        apply_renames(plans, dry_run=False, force=True)

        assert existing.read_bytes() == b"new"

    def test_two_plans_never_collide_on_one_target(self, tmp_path):
        first = tmp_path / "t00.mkv"
        second = tmp_path / "t01.mkv"
        first.write_bytes(b"a")
        second.write_bytes(b"b")
        plans = plan_renames(
            [result_for(first, 1, "X"), result_for(second, 1, "X")], series="Show"
        )

        outcomes = apply_renames(plans, dry_run=False)

        assert sum(outcome.applied for outcome in outcomes) == 1

    def test_a_missing_source_is_reported_not_raised(self, tmp_path):
        source = tmp_path / "gone.mkv"
        source.touch()
        plans = plan_renames([result_for(source, 1, "X")], series="Show")
        source.unlink()

        outcomes = apply_renames(plans, dry_run=False)

        assert outcomes[0].applied is False
        assert isinstance(outcomes[0], RenameOutcome)
