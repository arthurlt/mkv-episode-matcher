"""Tests for the report table, the JSON payload, and preview export."""

from __future__ import annotations

import json
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
from mkv_episode_matcher.report import (
    build_payload,
    export_previews,
    render_table,
    table_rows,
    write_json,
)

from .conftest import requires_ffmpeg


def video(name: str = "title_t00.mkv") -> VideoFile:
    return VideoFile(Path(name), duration_s=1320.0, size_bytes=10, mtime_ns=1)


def matched_result(**overrides) -> MatchResult:
    episode = Episode(season=2, number=1, title="Chipped Beef")
    hit = StillHit(
        still=Still("tmdb", "https://image.tmdb.org/t/p/w780/a.jpg"),
        distance=3,
        timestamp=612.5,
        dhash_distance=4,
    )
    defaults = dict(
        video=video(),
        status=MatchStatus.MATCHED,
        episode=episode,
        hit=hit,
        cost=3.0,
        runner_up=Episode(season=2, number=2, title="Blood Money"),
        runner_up_cost=28.0,
        supporting_stills=2,
    )
    defaults.update(overrides)
    return MatchResult(**defaults)


class TestPayload:
    def test_reports_the_matched_episode(self):
        payload = build_payload([matched_result()], series="Test Precinct", season=2)

        entry = payload["files"][0]
        assert entry["status"] == "matched"
        assert entry["episode"] == "S02E01"
        assert entry["title"] == "Chipped Beef"

    def test_includes_the_evidence_a_human_needs_to_spot_check(self):
        entry = build_payload([matched_result()], series="Test Precinct", season=2)["files"][0]

        assert entry["matched_timestamp"] == pytest.approx(612.5)
        assert entry["matched_timestamp_hms"] == "00:10:12"
        assert entry["still_url"].endswith("/a.jpg")
        assert entry["distance"] == 3
        assert entry["supporting_stills"] == 2

    def test_includes_the_runner_up_and_gap(self):
        entry = build_payload([matched_result()], series="Test Precinct", season=2)["files"][0]

        assert entry["runner_up"] == "S02E02"
        assert entry["gap"] == pytest.approx(25.0)

    def test_records_the_series_and_season(self):
        payload = build_payload([matched_result()], series="Test Precinct", season=2)

        assert payload["series"] == "Test Precinct"
        assert payload["season"] == 2

    def test_summarises_the_statuses(self):
        results = [
            matched_result(),
            MatchResult(video=video("b.mkv"), status=MatchStatus.AMBIGUOUS),
            MatchResult(video=video("c.mkv"), status=MatchStatus.UNMATCHED),
            MatchResult(
                video=video("d.mkv"), status=MatchStatus.SKIPPED, skip_reason=SkipReason.TOO_SHORT
            ),
        ]

        summary = build_payload(results, series="X", season=2)["summary"]

        assert summary == {"matched": 1, "ambiguous": 1, "unmatched": 1, "skipped": 1}

    def test_an_unmatched_file_has_null_episode_fields(self):
        payload = build_payload(
            [MatchResult(video=video(), status=MatchStatus.UNMATCHED)], series="X", season=2
        )

        entry = payload["files"][0]
        assert entry["episode"] is None
        assert entry["matched_timestamp"] is None

    def test_notes_are_carried_through(self):
        result = MatchResult(
            video=video(), status=MatchStatus.AMBIGUOUS, notes=["two episodes are close"]
        )

        entry = build_payload([result], series="X", season=2)["files"][0]

        assert entry["notes"] == ["two episodes are close"]

    def test_payload_is_json_serialisable(self, tmp_path):
        target = tmp_path / "report.json"

        write_json(target, build_payload([matched_result()], series="X", season=2))

        assert json.loads(target.read_text())["files"][0]["episode"] == "S02E01"

    def test_write_json_creates_missing_directories(self, tmp_path):
        target = tmp_path / "nested" / "report.json"

        write_json(target, {"a": 1})

        assert target.exists()


class TestTable:
    def test_renders_a_row_per_file(self):
        table = render_table([matched_result(), MatchResult(video=video("b.mkv"), status=MatchStatus.UNMATCHED)])

        assert table.row_count == 2

    def test_shows_the_timestamp_and_episode(self):
        row = table_rows([matched_result()])[0]

        assert row[0] == "title_t00.mkv"
        assert row[2] == "S02E01"
        assert row[6] == "00:10:12"

    def test_shows_why_a_result_is_ambiguous(self):
        result = MatchResult(
            video=video(), status=MatchStatus.AMBIGUOUS, notes=["too close to call"]
        )

        assert table_rows([result])[0][-1] == "too close to call"

    def test_placeholders_stand_in_for_missing_values(self):
        row = table_rows([MatchResult(video=video(), status=MatchStatus.UNMATCHED)])[0]

        assert row[2] == "-"
        assert row[6] == "-"

    def test_handles_an_empty_result_set(self):
        assert render_table([]).row_count == 0
        assert table_rows([]) == []

    def test_every_row_has_one_cell_per_column(self):
        from mkv_episode_matcher.report import TABLE_COLUMNS

        rows = table_rows([matched_result(), MatchResult(video=video("b.mkv"), status=MatchStatus.SKIPPED)])

        assert all(len(row) == len(TABLE_COLUMNS) for row in rows)


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestPreviews:
    def test_exports_a_side_by_side_image_for_a_match(self, video_factory, tmp_path, cache_dir):
        from PIL import Image

        from .conftest import make_pattern

        path = video_factory("clip.mkv", [1, 2, 3, 4], seconds_per_frame=1.0)
        still_path = cache_dir / "still.png"
        Image.fromarray(make_pattern(3)).save(still_path)
        result = matched_result(
            video=VideoFile(path, 4.0, path.stat().st_size, path.stat().st_mtime_ns),
            hit=StillHit(Still("tmdb", "https://img/a.jpg"), 1, 2.0, 1),
        )

        written = export_previews(
            [result], still_paths={"https://img/a.jpg": still_path}, destination=tmp_path / "previews"
        )

        assert len(written) == 1
        with Image.open(written[0]) as preview:
            assert preview.width > preview.height

    def test_skips_results_without_a_hit(self, tmp_path):
        written = export_previews(
            [MatchResult(video=video(), status=MatchStatus.UNMATCHED)],
            still_paths={},
            destination=tmp_path / "previews",
        )

        assert written == []

    def test_skips_when_the_still_is_not_on_disk(self, video_factory, tmp_path):
        path = video_factory("clip.mkv", [1, 2])
        result = matched_result(
            video=VideoFile(path, 2.0, 1, 1),
            hit=StillHit(Still("tmdb", "https://img/gone.jpg"), 1, 0.5, 1),
        )

        assert export_previews([result], still_paths={}, destination=tmp_path / "p") == []
