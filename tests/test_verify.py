"""Tests for NCC verification and aspect-aware still/frame alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mkv_episode_matcher.models import (
    Episode,
    EpisodeScore,
    FileScores,
    Still,
    StillHit,
    VideoFile,
)
from mkv_episode_matcher.verify import (
    apply_still_uniqueness,
    aspect_aligned_pairs,
    best_ncc,
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
    def test_shared_still_is_demoted_across_files(self):
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

        assert demoted[0].scores[0].ncc == pytest.approx(0.40)
        assert demoted[1].scores[0].ncc == pytest.approx(0.35)
        assert demoted[0].scores[0].cost == pytest.approx(0.60)

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
