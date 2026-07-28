"""Tests for NCC verification and aspect-aware still/frame alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mkv_episode_matcher.models import (
    Episode,
    EpisodeScore,
    FileScores,
    MatchResult,
    MatchStatus,
    Still,
    StillHit,
    VideoFile,
)
from mkv_episode_matcher.verify import (
    apply_duration_corroboration,
    apply_still_uniqueness,
    aspect_aligned_pairs,
    best_ncc,
    match_by_unique_duration,
    ncc,
)


def _gradient(height: int, width: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, width, dtype=np.float32)
    rows = np.tile(base, (height, 1))
    noise = rng.normal(0, 8, size=rows.shape)
    return np.clip(rows + noise, 0, 255).astype(np.uint8)


class TestNcc:
    def test_identical_images_score_one(self):
        image = _gradient(64, 96, seed=1)

        assert ncc(image, image) == pytest.approx(1.0, abs=1e-5)

    def test_inverted_images_score_negative_one(self):
        image = _gradient(64, 96, seed=2)

        assert ncc(image, 255 - image) == pytest.approx(-1.0, abs=1e-5)

    def test_unrelated_images_score_low(self):
        left = _gradient(64, 96, seed=3)
        right = np.random.default_rng(99).integers(0, 256, size=(64, 96), dtype=np.uint8)

        assert abs(ncc(left, right)) < 0.35


class TestAspectAlignedPairs:
    def test_includes_the_uncropped_pair(self):
        still = _gradient(45, 80)  # 16:9-ish
        frame = _gradient(40, 80)  # 2:1

        pairs = aspect_aligned_pairs(still, frame)

        assert pairs[0][0].shape == still.shape
        assert pairs[0][1].shape == frame.shape
        assert len(pairs) > 1

    def test_crop_search_recovers_a_vertically_cropped_still(self):
        """A 2:1 frame that is a vertical crop of a 16:9 still should NCC well."""
        still = _gradient(90, 160, seed=7)
        # Frame is the vertical center crop to 2:1.
        target_h = round(160 / 2.0)
        y0 = (90 - target_h) // 2
        frame = still[y0 : y0 + target_h, :]

        assert best_ncc(still, frame) > 0.95
        assert best_ncc(still, frame) >= ncc(still, frame) - 1e-9


class TestStillUniqueness:
    def test_weaker_shared_still_claim_is_dropped_stronger_kept(self):
        still = Still("tmdb", "https://img/shared.jpg")
        episode = Episode(season=2, number=1, title="One", stills=(still,))

        def file_score(name: str, value: float) -> FileScores:
            hit = StillHit(still=still, distance=10, timestamp=1.0, dhash_distance=2)
            return FileScores(
                video=VideoFile(Path(name), 100.0, 1, 1),
                scores=(
                    EpisodeScore(
                        episode=episode,
                        best_hit=hit,
                        supporting_stills=1,
                        cost=1.0 - value,
                        ncc=value,
                    ),
                ),
            )

        demoted = apply_still_uniqueness(
            [file_score("a.mkv", 0.80), file_score("b.mkv", 0.70)], floor=0.5
        )

        assert demoted[0].scores[0].ncc == pytest.approx(0.80)
        assert demoted[0].scores[0].cost == pytest.approx(0.20)
        assert demoted[1].scores[0].ncc is None
        assert demoted[1].scores[0].cost == float("inf")

    def test_raising_candidates_must_not_erase_a_clear_winner(self):
        """A soft false peak on the same still must not halve a strong match."""
        still = Still("tmdb", "https://img/e01.jpg")
        episode = Episode(season=2, number=1, title="One", stills=(still,))

        def file_score(name: str, value: float, supporting: int = 0) -> FileScores:
            hit = StillHit(still=still, distance=16, timestamp=120.0, dhash_distance=22)
            return FileScores(
                video=VideoFile(Path(name), 100.0, 1, 1),
                scores=(
                    EpisodeScore(
                        episode=episode,
                        best_hit=hit,
                        supporting_stills=supporting,
                        cost=1.0 - value,
                        ncc=value,
                    ),
                ),
            )

        demoted = apply_still_uniqueness(
            [file_score("E1.mkv", 0.837, supporting=6), file_score("E2.mkv", 0.55)],
            floor=0.5,
        )

        assert demoted[0].scores[0].ncc == pytest.approx(0.837)
        assert demoted[0].scores[0].ncc >= 0.65
        assert demoted[1].scores[0].ncc is None

    def test_soft_shared_claims_are_not_locked(self):
        """Gray-zone peaks must not starve other files of the same still."""
        still = Still("tmdb", "https://img/e09.jpg")
        episode = Episode(season=2, number=9, title="Beard", stills=(still,))

        def file_score(name: str, value: float) -> FileScores:
            hit = StillHit(still=still, distance=18, timestamp=100.0, dhash_distance=20)
            return FileScores(
                video=VideoFile(Path(name), 100.0, 1, 1),
                scores=(
                    EpisodeScore(
                        episode=episode,
                        best_hit=hit,
                        supporting_stills=0,
                        cost=1.0 - value,
                        ncc=value,
                    ),
                ),
            )

        demoted = apply_still_uniqueness(
            [file_score("t01.mkv", 0.545), file_score("t02.mkv", 0.510)],
            floor=0.5,
            lock_threshold=0.65,
        )

        assert demoted[0].scores[0].ncc == pytest.approx(0.545)
        assert demoted[1].scores[0].ncc == pytest.approx(0.510)

    def test_unique_still_is_unchanged(self):
        still = Still("tmdb", "https://img/unique.jpg")
        episode = Episode(season=2, number=1, title="One", stills=(still,))
        hit = StillHit(still=still, distance=4, timestamp=1.0, dhash_distance=2)
        scored = [
            FileScores(
                video=VideoFile(Path("a.mkv"), 100.0, 1, 1),
                scores=(
                    EpisodeScore(
                        episode=episode,
                        best_hit=hit,
                        supporting_stills=1,
                        cost=0.2,
                        ncc=0.8,
                    ),
                ),
            )
        ]

        assert apply_still_uniqueness(scored)[0].scores[0].ncc == pytest.approx(0.8)


class TestDurationCorroboration:
    def test_boosts_matching_runtime_and_penalises_mismatch(self):
        still = Still("tmdb", "https://img/a.jpg")
        match_ep = Episode(season=2, number=10, title="A", stills=(still,), runtime_minutes=46)
        other_ep = Episode(season=2, number=9, title="B", stills=(still,), runtime_minutes=43)
        hit = StillHit(still=still, distance=18, timestamp=1.0, dhash_distance=18)
        scored = [
            FileScores(
                video=VideoFile(Path("t01.mkv"), 2769.0, 1, 1),
                scores=(
                    EpisodeScore(
                        episode=match_ep, best_hit=hit, supporting_stills=0, cost=0.375, ncc=0.625
                    ),
                    EpisodeScore(
                        episode=other_ep, best_hit=hit, supporting_stills=0, cost=0.455, ncc=0.545
                    ),
                ),
            )
        ]

        adjusted = apply_duration_corroboration(scored)[0]

        assert adjusted.scores[0].ncc == pytest.approx(0.745)
        assert adjusted.scores[1].ncc == pytest.approx(0.345)


class TestUniqueDurationFallback:
    def test_fills_unmatched_when_runtimes_are_a_unique_bijection(self):
        episodes = [
            Episode(season=2, number=8, title="Man City", runtime_minutes=46),
            Episode(season=2, number=9, title="Beard", runtime_minutes=43),
        ]
        results = [
            MatchResult(
                video=VideoFile(Path("t03.mkv"), 2730.0, 1, 1),
                status=MatchStatus.UNMATCHED,
                notes=["no still verified above the NCC threshold"],
            ),
            MatchResult(
                video=VideoFile(Path("t02.mkv"), 2585.0, 1, 1),
                status=MatchStatus.UNMATCHED,
                notes=["no still verified above the NCC threshold"],
            ),
        ]

        filled = match_by_unique_duration(results, episodes)

        assert all(result.status is MatchStatus.MATCHED for result in filled)
        by_name = {result.video.name: result.episode.number for result in filled}
        assert by_name == {"t03.mkv": 8, "t02.mkv": 9}
        assert "unique runtime" in filled[0].notes[0]

    def test_fills_ambiguous_leftovers_after_soft_visual_collisions(self):
        """Disc 2 regression: t00/t04 stayed ambiguous while wanting stolen episodes."""
        episodes = [
            Episode(season=2, number=11, title="Midnight Train", runtime_minutes=42),
            Episode(season=2, number=12, title="Pyramid", runtime_minutes=49),
        ]
        results = [
            MatchResult(
                video=VideoFile(Path("t00.mkv"), 2996.118, 1, 1),
                status=MatchStatus.AMBIGUOUS,
                ncc=0.56,
                notes=["another file is a better fit for S02E10"],
            ),
            MatchResult(
                video=VideoFile(Path("t01.mkv"), 2769.141, 1, 1),
                status=MatchStatus.MATCHED,
                episode=Episode(season=2, number=10, title="No Weddings", runtime_minutes=46),
                ncc=0.625,
            ),
            MatchResult(
                video=VideoFile(Path("t04.mkv"), 2546.71, 1, 1),
                status=MatchStatus.AMBIGUOUS,
                ncc=0.522,
                notes=["another file is a better fit for S02E09"],
            ),
        ]

        filled = match_by_unique_duration(results, episodes)
        by_name = {
            result.video.name: (
                result.status,
                None if result.episode is None else result.episode.number,
            )
            for result in filled
        }
        assert by_name["t00.mkv"] == (MatchStatus.MATCHED, 12)
        assert by_name["t04.mkv"] == (MatchStatus.MATCHED, 11)
        assert by_name["t01.mkv"] == (MatchStatus.MATCHED, 10)
