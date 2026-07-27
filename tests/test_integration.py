"""End-to-end tests over real videos with planted frames and replayed APIs.

The fixture is deliberately hostile to every shortcut the tool is forbidden
from taking: all four episodes have identical runtimes, and the disc title
order is a shuffle of the episode order. A correct answer can only come from
finding the provider stills inside the video.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mkv_episode_matcher import cli as cli_module
from mkv_episode_matcher.cli import app

from .conftest import encode_video, make_pattern, requires_ffmpeg, still_bytes

pytestmark = [requires_ffmpeg, pytest.mark.ffmpeg, pytest.mark.slow]

TMDB_PREFIX = "https://image.tmdb.org/t/p/w780"
TVDB_PREFIX = "https://artworks.thetvdb.com/banners/episodes/70328"

#: Every still the recorded fixtures expose, and the pattern seed it depicts.
STILL_SEEDS = {
    f"{TMDB_PREFIX}/s02e01-a.jpg": 2001,
    f"{TMDB_PREFIX}/s02e01-b.jpg": 2002,
    f"{TMDB_PREFIX}/s02e01-c.jpg": 2003,
    f"{TVDB_PREFIX}/301.jpg": 2004,
    f"{TVDB_PREFIX}/301-2.jpg": 2005,
    f"{TMDB_PREFIX}/s02e02-a.jpg": 2011,
    f"{TVDB_PREFIX}/302.jpg": 2012,
    f"{TMDB_PREFIX}/s02e03-a.jpg": 2021,
    f"{TMDB_PREFIX}/s02e03-b.jpg": 2022,
    f"{TVDB_PREFIX}/304.jpg": 2031,
    f"{TVDB_PREFIX}/304-2.jpg": 2032,
}

#: Which seeds are actually planted in each episode's rip. ``s02e03-b`` is
#: deliberately absent: a published still that appears nowhere in the episode.
PLANTED = {
    1: [2001, 2002, 2003, 2004, 2005],
    2: [2011, 2012],
    3: [2021],
    4: [2031, 2032],
}

#: Disc title order deliberately disagrees with episode order.
DISC_LAYOUT = {
    "title_t00.mkv": 3,
    "title_t01.mkv": 1,
    "title_t02.mkv": 4,
    "title_t03.mkv": 2,
}

SLOTS_PER_TITLE = 10
SECONDS_PER_SLOT = 2.0


@pytest.fixture
def disc(tmp_path: Path, recorded_api) -> Path:
    """Build a folder of equal-length rips with provider stills planted inside."""
    rips = tmp_path / "rips"
    rips.mkdir()

    for url, seed in STILL_SEEDS.items():
        recorded_api.image_bytes[url] = still_bytes(seed)

    for position, (name, episode_number) in enumerate(DISC_LAYOUT.items()):
        planted = PLANTED[episode_number]
        filler = [9000 + position * 100 + slot for slot in range(SLOTS_PER_TITLE)]
        seeds = list(filler)
        for index, seed in enumerate(planted):
            seeds[_planting_slot(index, len(planted))] = seed
        encode_video(
            rips / name,
            [make_pattern(seed) for seed in seeds],
            seconds_per_frame=SECONDS_PER_SLOT,
        )

    encode_video(
        rips / "title_t04.mkv",
        [make_pattern(8000 + index) for index in range(3)],
        seconds_per_frame=1.0,
    )
    return rips


def _planting_slot(index: int, total: int) -> int:
    """Spread planted frames evenly through the runtime, away from the edges."""
    return 1 + index * (SLOTS_PER_TITLE - 2) // max(1, total)


@pytest.fixture
def invoke(disc, recorded_api, cache_dir, monkeypatch):
    """Return a callable running the CLI against the planted disc."""
    monkeypatch.setattr(cli_module, "build_http_client", recorded_api.client)
    runner = CliRunner()

    def run(*extra: str):
        return runner.invoke(
            app,
            [
                "match",
                str(disc),
                "--series",
                "Test Precinct",
                "--season",
                "2",
                "--tmdb-key",
                "test-key",
                "--tvdb-key",
                "test-key",
                "--cache",
                str(cache_dir),
                "--interval",
                "1.0",
                "--min-minutes",
                "0.1",
                *extra,
            ],
            catch_exceptions=False,
        )

    return run


def read_report(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {entry["file"]: entry for entry in payload["files"]}


class TestEndToEnd:
    def test_every_rip_is_matched_to_the_right_episode(self, invoke, tmp_path):
        report = tmp_path / "report.json"

        result = invoke("--json", str(report))

        assert result.exit_code == 0
        entries = read_report(report)
        for name, episode_number in DISC_LAYOUT.items():
            assert entries[name]["status"] == "matched", entries[name]
            assert entries[name]["episode"] == f"S02E{episode_number:02d}"

    def test_precision_at_one_is_total_on_the_fixture(self, invoke, tmp_path):
        report = tmp_path / "report.json"
        invoke("--json", str(report))

        entries = read_report(report)
        correct = sum(
            entries[name]["episode"] == f"S02E{number:02d}" for name, number in DISC_LAYOUT.items()
        )

        assert correct / len(DISC_LAYOUT) == 1.0

    def test_disc_order_would_have_given_the_wrong_answer(self):
        """Guard the fixture itself: sorting by title must not solve the puzzle."""
        by_title_order = [DISC_LAYOUT[name] for name in sorted(DISC_LAYOUT)]

        assert by_title_order != sorted(by_title_order)

    def test_identical_runtimes_cannot_have_helped(self, invoke, tmp_path):
        report = tmp_path / "report.json"
        invoke("--json", str(report))

        entries = read_report(report)
        runtimes = {entries[name]["duration_s"] for name in DISC_LAYOUT}

        assert max(runtimes) - min(runtimes) < 0.5

    def test_the_menu_title_is_skipped_on_duration(self, invoke, tmp_path):
        report = tmp_path / "report.json"
        invoke("--json", str(report))

        entry = read_report(report)["title_t04.mkv"]

        assert entry["status"] == "skipped"
        assert entry["skip_reason"] == "too_short"

    def test_each_match_reports_a_still_and_a_timestamp_to_check(self, invoke, tmp_path):
        report = tmp_path / "report.json"
        invoke("--json", str(report))

        for name in DISC_LAYOUT:
            entry = read_report(report)[name]
            assert entry["still_url"] in STILL_SEEDS
            assert 0.0 <= entry["matched_timestamp"] <= SLOTS_PER_TITLE * SECONDS_PER_SLOT
            assert entry["matched_timestamp_hms"].count(":") == 2

    def test_the_matched_timestamp_lands_on_the_planted_frame(self, invoke, tmp_path):
        report = tmp_path / "report.json"
        invoke("--json", str(report))

        for name, episode_number in DISC_LAYOUT.items():
            entry = read_report(report)[name]
            seed = STILL_SEEDS[entry["still_url"]]
            slot = _planting_slot(PLANTED[episode_number].index(seed), len(PLANTED[episode_number]))
            planted_at = slot * SECONDS_PER_SLOT

            assert planted_at - 0.5 <= entry["matched_timestamp"] <= planted_at + SECONDS_PER_SLOT

    def test_an_episode_with_several_stills_gathers_multi_still_agreement(self, invoke, tmp_path):
        report = tmp_path / "report.json"
        invoke("--json", str(report))

        entries = read_report(report)
        multi = next(name for name, number in DISC_LAYOUT.items() if number == 1)
        single = next(name for name, number in DISC_LAYOUT.items() if number == 3)

        assert entries[multi]["supporting_stills"] >= 4
        assert entries[single]["supporting_stills"] == 1

    def test_summary_counts_are_reported(self, invoke, tmp_path):
        report = tmp_path / "report.json"
        invoke("--json", str(report))

        summary = json.loads(report.read_text())["summary"]

        assert summary["matched"] == 4
        assert summary["skipped"] == 1
        assert summary["ambiguous"] == 0

    def test_nothing_is_renamed_without_the_flag(self, invoke, disc):
        invoke()

        assert sorted(path.name for path in disc.glob("*.mkv")) == sorted(
            [*DISC_LAYOUT, "title_t04.mkv"]
        )

    def test_rename_produces_plex_style_names(self, invoke, disc):
        invoke("--rename")

        names = {path.name for path in disc.glob("*.mkv")}
        assert "Test Precinct - S02E01 - Chipped Beef.mkv" in names
        assert "Test Precinct - S02E03 - The Second Oldest Profession.mkv" in names
        assert "title_t04.mkv" in names

    def test_previews_are_exported_for_matches(self, invoke, tmp_path):
        previews = tmp_path / "previews"

        invoke("--previews", str(previews))

        assert len(list(previews.glob("*.jpg"))) == len(DISC_LAYOUT)

    def test_the_second_run_reuses_the_frame_indexes(self, invoke, monkeypatch, tmp_path):
        from mkv_episode_matcher import frames as frames_module

        invoke()

        calls = []
        original = frames_module.extract_frame_hashes
        monkeypatch.setattr(
            frames_module,
            "extract_frame_hashes",
            lambda *args, **kwargs: (calls.append(args), original(*args, **kwargs))[1],
        )
        report = tmp_path / "second.json"
        invoke("--json", str(report))

        assert calls == []
        assert json.loads(report.read_text())["summary"]["matched"] == 4

    def test_the_second_run_makes_no_network_calls(self, invoke, recorded_api):
        invoke()
        after_first = len(recorded_api.requests)

        invoke()

        assert after_first > 0
        assert len(recorded_api.requests) == after_first

    def test_refinement_sharpens_a_coarse_pass(self, invoke, tmp_path):
        coarse_report = tmp_path / "coarse.json"
        refined_report = tmp_path / "refined.json"

        invoke("--interval", "3.0", "--json", str(coarse_report))
        assert invoke("--interval", "3.0", "--refine", "--json", str(refined_report)).exit_code == 0

        coarse = read_report(coarse_report)
        refined = read_report(refined_report)
        for name, number in DISC_LAYOUT.items():
            assert refined[name]["episode"] == f"S02E{number:02d}"
            assert refined[name]["distance"] <= coarse[name]["distance"]

    def test_sampling_too_coarsely_to_see_a_still_never_invents_a_match(self, invoke, tmp_path):
        """A still that falls between samples must cost a match, not cause a wrong one."""
        report = tmp_path / "coarse.json"

        invoke("--interval", "4.0", "--json", str(report))

        entries = read_report(report)
        for name, number in DISC_LAYOUT.items():
            if entries[name]["status"] == "matched":
                assert entries[name]["episode"] == f"S02E{number:02d}"
            else:
                assert entries[name]["status"] in {"ambiguous", "unmatched"}

    def test_strict_mode_succeeds_when_everything_resolves(self, invoke):
        assert invoke("--strict").exit_code == 0


class TestDegradedRuns:
    def test_without_thetvdb_the_tmdb_only_episode_is_unmatched(self, invoke, tmp_path):
        """Episode 4 has no TMDB stills, so nothing can identify its rip."""
        report = tmp_path / "report.json"

        invoke("--no-tvdb", "--json", str(report))

        entries = read_report(report)
        blind_title = next(name for name, number in DISC_LAYOUT.items() if number == 4)
        assert entries[blind_title]["status"] == "unmatched"
        assert entries[blind_title]["episode"] is None

    def test_a_blind_episode_never_gets_renamed(self, invoke, disc):
        invoke("--no-tvdb", "--rename")

        blind_title = next(name for name, number in DISC_LAYOUT.items() if number == 4)
        assert (disc / blind_title).exists()

    def test_strict_mode_reports_unresolved_files(self, invoke):
        assert invoke("--no-tvdb", "--strict").exit_code == 2

    def test_planted_frames_are_recovered_exactly(self, invoke, tmp_path):
        """Encoding, rescaling and JPEG round-tripping must not perturb the hash."""
        report = tmp_path / "report.json"

        invoke("--threshold", "0", "--json", str(report))

        entries = read_report(report)
        assert all(entries[name]["distance"] == 0 for name in DISC_LAYOUT)
        assert json.loads(report.read_text())["summary"]["matched"] == 4

    def test_demanding_an_impossible_gap_makes_everything_ambiguous(self, invoke, tmp_path):
        report = tmp_path / "report.json"

        invoke("--gap", "1000", "--json", str(report))

        summary = json.loads(report.read_text())["summary"]
        assert summary["matched"] == 0
        assert summary["ambiguous"] == 4

    def test_missing_credentials_are_reported_clearly(self, disc, cache_dir, monkeypatch):
        monkeypatch.delenv("TMDB_API_KEY", raising=False)
        monkeypatch.delenv("TVDB_API_KEY", raising=False)
        runner = CliRunner()

        result = runner.invoke(
            app,
            ["match", str(disc), "--series", "X", "--season", "2", "--cache", str(cache_dir)],
        )

        assert result.exit_code != 0
        assert "TMDB_API_KEY" in result.output

    def test_both_providers_disabled_is_rejected(self, disc, cache_dir):
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "match",
                str(disc),
                "--series",
                "X",
                "--season",
                "2",
                "--cache",
                str(cache_dir),
                "--no-tmdb",
                "--no-tvdb",
            ],
        )

        assert result.exit_code != 0


class TestInterruption:
    def _invoke_raising(self, disc, cache_dir, monkeypatch, error):
        def explode(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(cli_module, "run_match", explode)
        return CliRunner().invoke(
            app,
            [
                "match",
                str(disc),
                "--series",
                "Test Precinct",
                "--season",
                "2",
                "--tmdb-key",
                "k",
                "--tvdb-key",
                "k",
                "--cache",
                str(cache_dir),
                "--min-minutes",
                "0.1",
            ],
        )

    def test_a_cancelled_run_exits_with_the_conventional_code(
        self, disc, cache_dir, monkeypatch, recorded_api
    ):
        from mkv_episode_matcher.cancellation import OperationCancelledError

        monkeypatch.setattr(cli_module, "build_http_client", recorded_api.client)

        result = self._invoke_raising(
            disc, cache_dir, monkeypatch, OperationCancelledError("cancelled")
        )

        assert result.exit_code == 130

    def test_a_cancelled_run_says_what_happened_instead_of_a_traceback(
        self, disc, cache_dir, monkeypatch, recorded_api
    ):
        from mkv_episode_matcher.cancellation import OperationCancelledError

        monkeypatch.setattr(cli_module, "build_http_client", recorded_api.client)

        result = self._invoke_raising(
            disc, cache_dir, monkeypatch, OperationCancelledError("cancelled")
        )

        assert "Interrupted" in result.output
        assert "Traceback" not in result.output

    def test_a_bare_keyboard_interrupt_also_exits_with_130(
        self, disc, cache_dir, monkeypatch, recorded_api
    ):
        monkeypatch.setattr(cli_module, "build_http_client", recorded_api.client)

        result = self._invoke_raising(disc, cache_dir, monkeypatch, KeyboardInterrupt())

        assert result.exit_code == 130

    def test_a_normal_run_is_unaffected(self, invoke):
        assert invoke().exit_code == 0


class TestClearCache:
    def test_removes_the_cached_indexes(self, invoke, cache_dir):
        invoke()
        assert list((cache_dir / "frames").glob("*.npz"))

        CliRunner().invoke(app, ["clear-cache", "--cache", str(cache_dir), "--frames-only"])

        assert not (cache_dir / "frames").exists()
