"""Tests for the still-to-frame-index search and the scoring rules."""

from __future__ import annotations

import io

import httpx
import numpy as np
import pytest
from PIL import Image

from mkv_episode_matcher.cache import ImageCache, JsonCache
from mkv_episode_matcher.frames import FrameIndex, IndexParams
from mkv_episode_matcher.models import Episode, Still, StillKind, VideoFile
from mkv_episode_matcher.normalize import ImageHashes, hash_array
from mkv_episode_matcher.score_visual import (
    ScoringConfig,
    episode_cost,
    hash_stills,
    score_episode,
    score_file,
    score_all,
)

from .conftest import make_pattern, still_bytes


def index_of(hashes: list[ImageHashes], *, step: float = 1.0) -> FrameIndex:
    return FrameIndex(
        video_path=__import__("pathlib").Path("t00.mkv"),
        params=IndexParams(),
        timestamps=np.arange(len(hashes), dtype=np.float32) * step,
        phashes=np.asarray([h.phash for h in hashes], dtype=np.uint64),
        dhashes=np.asarray([h.dhash for h in hashes], dtype=np.uint64),
        source_size=1,
        source_mtime_ns=1,
    )


def episode_with(number: int, seeds: list[int]) -> Episode:
    return Episode(
        season=2,
        number=number,
        title=f"Episode {number}",
        stills=tuple(
            Still(provider="tmdb", url=f"https://img/{number}-{seed}.jpg") for seed in seeds
        ),
    )


def hashes_for(seeds: list[int]) -> dict[str, ImageHashes]:
    return {
        f"https://img/{number}-{seed}.jpg": hash_array(make_pattern(seed))
        for number in range(1, 10)
        for seed in seeds
    }


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "title_t00.mkv"
    path.write_bytes(b"x")
    return VideoFile(path, 1320.0, 1, 1)


class TestEpisodeCost:
    def test_cost_is_the_best_distance_when_a_single_still_hits(self):
        assert episode_cost(6, supporting_stills=1, config=ScoringConfig()) == pytest.approx(6.0)

    def test_multiple_agreeing_stills_lower_the_cost(self):
        config = ScoringConfig()

        assert episode_cost(6, supporting_stills=3, config=config) < episode_cost(
            6, supporting_stills=1, config=config
        )

    def test_the_agreement_bonus_is_capped(self):
        config = ScoringConfig(agreement_bonus=1.0, max_agreement_bonus=2.0)

        assert episode_cost(10, supporting_stills=50, config=config) == pytest.approx(8.0)

    def test_cost_never_goes_negative(self):
        config = ScoringConfig(agreement_bonus=5.0, max_agreement_bonus=100.0)

        assert episode_cost(1, supporting_stills=10, config=config) >= 0.0

    def test_a_missing_hit_costs_infinity(self):
        assert episode_cost(None, supporting_stills=0, config=ScoringConfig()) == float("inf")


class TestScoreEpisode:
    def test_finds_a_planted_still_and_its_timestamp(self):
        frames = [hash_array(make_pattern(seed)) for seed in (10, 11, 12, 13)]
        episode = episode_with(1, [12])

        score = score_episode(index_of(frames, step=2.0), episode, hashes_for([12]), ScoringConfig())

        assert score.best_distance == 0
        assert score.best_hit.timestamp == pytest.approx(4.0)
        assert score.best_hit.still.url.endswith("1-12.jpg")

    def test_counts_every_still_that_clears_the_threshold(self):
        frames = [hash_array(make_pattern(seed)) for seed in (20, 21, 22, 23)]
        episode = episode_with(1, [21, 22])

        score = score_episode(index_of(frames), episode, hashes_for([21, 22]), ScoringConfig())

        assert score.supporting_stills == 2

    def test_stills_that_are_not_in_the_file_do_not_support_it(self):
        frames = [hash_array(make_pattern(seed)) for seed in (30, 31)]
        episode = episode_with(1, [31, 99])

        score = score_episode(index_of(frames), episode, hashes_for([31, 99]), ScoringConfig())

        assert score.supporting_stills == 1
        assert score.best_distance == 0

    def test_an_episode_with_no_stills_scores_infinite_cost(self):
        frames = [hash_array(make_pattern(40))]
        episode = Episode(season=2, number=1, title="Blind", stills=())

        score = score_episode(index_of(frames), episode, {}, ScoringConfig())

        assert score.best_hit is None
        assert score.cost == float("inf")

    def test_a_still_that_failed_to_download_is_ignored(self):
        frames = [hash_array(make_pattern(50))]
        episode = episode_with(1, [50])

        score = score_episode(index_of(frames), episode, {}, ScoringConfig())

        assert score.best_hit is None

    def test_an_empty_index_produces_no_hit(self):
        episode = episode_with(1, [60])

        score = score_episode(index_of([]), episode, hashes_for([60]), ScoringConfig())

        assert score.best_hit is None

    def test_a_dhash_disagreement_penalises_a_suspicious_phash_hit(self):
        frame = hash_array(make_pattern(70))
        colliding = ImageHashes(phash=frame.phash, dhash=frame.dhash ^ 0xFFFFFFFFFFFFFFFF)
        episode = episode_with(1, [70])
        lookup = {f"https://img/1-70.jpg": colliding}

        score = score_episode(index_of([frame]), episode, lookup, ScoringConfig())

        assert score.best_distance == 0
        assert score.supporting_stills == 0
        assert score.cost > 0


class TestScoreFile:
    def test_ranks_the_episode_whose_still_is_present_first(self):
        frames = [hash_array(make_pattern(seed)) for seed in (80, 81, 82)]
        episodes = [episode_with(1, [200]), episode_with(2, [81]), episode_with(3, [201])]
        lookup = hashes_for([200, 81, 201])

        scores = score_file(
            VideoFile(__import__("pathlib").Path("a.mkv"), 1320.0, 1, 1),
            index_of(frames),
            episodes,
            lookup,
            ScoringConfig(),
        )

        assert scores.ranked()[0].episode.number == 2
        assert scores.ranked()[0].best_distance == 0

    def test_scores_every_episode_even_when_nothing_matches(self, video):
        frames = [hash_array(make_pattern(90))]
        episodes = [episode_with(number, [500 + number]) for number in (1, 2, 3)]

        scores = score_file(video, index_of(frames), episodes, hashes_for([501, 502, 503]), ScoringConfig())

        assert len(scores.scores) == 3

    def test_score_all_covers_every_indexed_file(self, tmp_path):
        from pathlib import Path

        videos = [VideoFile(Path(tmp_path / f"{n}.mkv"), 1320.0, 1, 1) for n in range(3)]
        indexes = {
            video.path: index_of([hash_array(make_pattern(100 + n))])
            for n, video in enumerate(videos)
        }
        episodes = [episode_with(n + 1, [100 + n]) for n in range(3)]

        results = score_all(videos, indexes, episodes, hashes_for([100, 101, 102]), ScoringConfig())

        assert len(results) == 3
        assert [result.ranked()[0].episode.number for result in results] == [1, 2, 3]

    def test_a_file_without_an_index_is_skipped(self, video):
        results = score_all([video], {}, [episode_with(1, [1])], {}, ScoringConfig())

        assert results == []


class TestHashStills:
    def test_downloads_and_hashes_each_still(self, recorded_api, cache_dir):
        stills = [Still("tmdb", "https://image.tmdb.org/t/p/w780/a.jpg")]
        recorded_api.image_bytes["https://image.tmdb.org/t/p/w780/a.jpg"] = still_bytes(5)

        hashed = hash_stills(
            stills, client=recorded_api.client(), images=ImageCache(cache_dir), hash_cache=None
        )

        assert set(hashed) == {stills[0].url}
        assert hashed[stills[0].url].phash == hash_array(make_pattern(5, 320, 180)).phash

    def test_a_failed_download_is_omitted(self, cache_dir):
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))

        hashed = hash_stills(
            [Still("tmdb", "https://img/a.jpg")],
            client=client,
            images=ImageCache(cache_dir),
            hash_cache=None,
        )

        assert hashed == {}

    def test_an_undecodable_image_is_omitted(self, cache_dir):
        client = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"not an image"))
        )

        hashed = hash_stills(
            [Still("tmdb", "https://img/a.jpg")],
            client=client,
            images=ImageCache(cache_dir),
            hash_cache=None,
        )

        assert hashed == {}

    def test_hashes_are_cached_between_runs(self, recorded_api, cache_dir):
        stills = [Still("tmdb", "https://image.tmdb.org/t/p/w780/a.jpg")]
        hash_cache = JsonCache(cache_dir / "hashes")

        first = hash_stills(
            stills, client=recorded_api.client(), images=ImageCache(cache_dir), hash_cache=hash_cache
        )
        (cache_dir / stills[0].filename).unlink()
        second = hash_stills(
            stills, client=recorded_api.client(), images=ImageCache(cache_dir), hash_cache=hash_cache
        )

        assert first == second

    def test_duplicate_urls_are_hashed_once(self, recorded_api, cache_dir):
        still = Still("tmdb", "https://image.tmdb.org/t/p/w780/a.jpg")

        hash_stills(
            [still, still], client=recorded_api.client(), images=ImageCache(cache_dir), hash_cache=None
        )

        assert recorded_api.paths_called("/a.jpg") == 1

    def test_a_greyscale_png_still_is_handled(self, cache_dir):
        buffer = io.BytesIO()
        Image.fromarray(make_pattern(6)).save(buffer, format="PNG")

        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, content=buffer.getvalue())
            )
        )
        hashed = hash_stills(
            [Still("tmdb", "https://img/a.png", kind=StillKind.SCREENCAP)],
            client=client,
            images=ImageCache(cache_dir),
            hash_cache=None,
        )

        assert len(hashed) == 1
