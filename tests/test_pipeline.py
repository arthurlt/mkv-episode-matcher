"""Tests for pipeline orchestration and the two-stage refinement pass."""

from __future__ import annotations

import pytest

from mkv_episode_matcher.cache import CacheRoot, ImageCache, JsonCache
from mkv_episode_matcher.frames import FrameIndexCache, FrameIndexer, IndexParams
from mkv_episode_matcher.models import Episode, FileScores, Still
from mkv_episode_matcher.normalize import hash_array
from mkv_episode_matcher.pipeline import MatchRequest, _refine, run_match
from mkv_episode_matcher.probe import DurationFilter, probe_video
from mkv_episode_matcher.score_visual import ScoringConfig, score_episode
from mkv_episode_matcher.tmdb_client import TmdbClient
from mkv_episode_matcher.tvdb_client import TvdbClient

from .conftest import make_pattern, requires_ffmpeg

SLOT_SECONDS = 0.5
TARGET_SLOT = 9
TARGET_SEED = 4109


@pytest.fixture
def fast_cut_video(video_factory):
    """A video whose picture changes twice a second, so coarse sampling misses shots."""
    seeds = [4100 + slot for slot in range(20)]
    seeds[TARGET_SLOT] = TARGET_SEED
    return video_factory("fast.mkv", seeds, seconds_per_frame=SLOT_SECONDS)


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestRefine:
    config = ScoringConfig()

    def _setup(self, fast_cut_video, cache_dir):
        video = probe_video(fast_cut_video)
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=4.0))
        index = indexer.get(video)
        episode = Episode(
            season=2,
            number=1,
            title="Target",
            stills=(Still("tmdb", "https://img/target.jpg"),),
        )
        hashes = {"https://img/target.jpg": hash_array(make_pattern(TARGET_SEED))}
        coarse = score_episode(index, episode, hashes, self.config)
        return video, indexer, hashes, coarse

    def test_the_coarse_pass_really_does_miss_the_target(self, fast_cut_video, cache_dir):
        """Guard the fixture: without this, the refinement test would prove nothing."""
        coarse = self._setup(fast_cut_video, cache_dir)[-1]

        assert coarse.best_distance > self.config.match_threshold

    def test_refinement_recovers_a_shot_between_coarse_samples(self, fast_cut_video, cache_dir):
        video, indexer, hashes, coarse = self._setup(fast_cut_video, cache_dir)
        scores = FileScores(video=video, scores=(coarse,))
        # A window wide enough to reach the target from any coarse sample, so the
        # test does not depend on which arbitrary frame the coarse pass settled on.
        request = _request(video, refine_window_s=6.0)

        refined = _refine(scores, indexer, hashes, request)

        best = refined.scores[0]
        assert best.cost < coarse.cost
        assert best.best_distance <= 4
        assert best.best_hit.timestamp == pytest.approx(TARGET_SLOT * SLOT_SECONDS, abs=0.3)

    def test_refinement_keeps_the_original_when_nothing_improves(self, video_factory, cache_dir):
        path = video_factory(
            "steady.mkv", [4200 + slot for slot in range(6)], seconds_per_frame=2.0
        )
        video = probe_video(path)
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=1.0))
        episode = Episode(
            season=2, number=1, title="T", stills=(Still("tmdb", "https://img/a.jpg"),)
        )
        hashes = {"https://img/a.jpg": hash_array(make_pattern(4202))}
        original = score_episode(indexer.get(video), episode, hashes, self.config)
        assert original.best_distance == 0

        refined = _refine(
            FileScores(video=video, scores=(original,)), indexer, hashes, _request(video)
        )

        assert refined.scores[0].cost == original.cost

    def test_refinement_does_nothing_without_a_hit(self, fast_cut_video, cache_dir):
        video = probe_video(fast_cut_video)
        indexer = FrameIndexer(FrameIndexCache(cache_dir), IndexParams(interval_s=4.0))
        blind = score_episode(
            indexer.get(video), Episode(season=2, number=1, title="T"), {}, self.config
        )
        scores = FileScores(video=video, scores=(blind,))

        assert _refine(scores, indexer, {}, _request(video)) is scores


def _request(video, *, refine_window_s: float = 3.0) -> MatchRequest:
    return MatchRequest(
        input_dir=video.path.parent,
        series="Test Precinct",
        season=2,
        index_params=IndexParams(interval_s=4.0),
        refine=True,
        refine_interval_s=0.25,
        refine_window_s=refine_window_s,
    )


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestRunMatchCancellation:
    def test_an_already_cancelled_run_stops_before_touching_anything(
        self, video_factory, cache_dir, recorded_api, tmp_path
    ):
        from mkv_episode_matcher.cancellation import CancellationToken, OperationCancelledError

        rips = tmp_path / "rips"
        video_factory("title_t00.mkv", list(range(8000, 8010)), directory=rips)
        cache = CacheRoot(cache_dir)
        http = recorded_api.client()
        token = CancellationToken()
        token.cancel()

        with pytest.raises(OperationCancelledError):
            run_match(
                MatchRequest(
                    input_dir=rips,
                    series="Test Precinct",
                    season=2,
                    tmdb_series_id="4224",
                    duration_filter=DurationFilter(min_duration_s=1.0),
                ),
                http=http,
                cache=cache,
                tmdb=TmdbClient(api_key="k", client=http, cache=JsonCache(cache.api_dir / "tmdb")),
                tvdb=None,
                token=token,
            )

        assert recorded_api.requests == []
        assert not list(cache.frames_dir.glob("*.npz"))


@requires_ffmpeg
@pytest.mark.ffmpeg
class TestRunMatch:
    def test_reports_metadata_alongside_the_verdicts(
        self, video_factory, cache_dir, recorded_api, tmp_path
    ):
        rips = tmp_path / "rips"
        video_factory("title_t00.mkv", list(range(5000, 5010)), directory=rips)
        cache = CacheRoot(cache_dir)
        http = recorded_api.client()

        run = run_match(
            MatchRequest(
                input_dir=rips,
                series="Test Precinct",
                season=2,
                tmdb_series_id="4224",
                tvdb_series_id="70328",
                duration_filter=DurationFilter(min_duration_s=1.0),
            ),
            http=http,
            cache=cache,
            tmdb=TmdbClient(api_key="k", client=http, cache=JsonCache(cache.api_dir / "tmdb")),
            tvdb=TvdbClient(api_key="k", client=http, cache=JsonCache(cache.api_dir / "tvdb")),
        )

        assert [episode.number for episode in run.episodes] == [1, 2, 3, 4]
        assert run.coverage.total_episodes == 4
        assert run.coverage.total_stills >= 6
        assert run.elapsed_s > 0

    def test_a_rip_containing_no_stills_is_unresolved(
        self, video_factory, cache_dir, recorded_api, tmp_path
    ):
        rips = tmp_path / "rips"
        video_factory("title_t00.mkv", list(range(6000, 6010)), directory=rips)
        cache = CacheRoot(cache_dir)
        http = recorded_api.client()

        run = run_match(
            MatchRequest(
                input_dir=rips,
                series="Test Precinct",
                season=2,
                tmdb_series_id="4224",
                duration_filter=DurationFilter(min_duration_s=1.0),
            ),
            http=http,
            cache=cache,
            tmdb=TmdbClient(api_key="k", client=http, cache=JsonCache(cache.api_dir / "tmdb")),
            tvdb=None,
        )

        assert len(run.unresolved) == 1
        assert run.results[0].episode is None

    def test_still_paths_point_at_downloaded_images(
        self, video_factory, cache_dir, recorded_api, tmp_path
    ):
        rips = tmp_path / "rips"
        video_factory("title_t00.mkv", list(range(7000, 7010)), directory=rips)
        cache = CacheRoot(cache_dir)
        http = recorded_api.client()

        run = run_match(
            MatchRequest(
                input_dir=rips,
                series="Test Precinct",
                season=2,
                tmdb_series_id="4224",
                duration_filter=DurationFilter(min_duration_s=1.0),
            ),
            http=http,
            cache=cache,
            tmdb=TmdbClient(api_key="k", client=http, cache=JsonCache(cache.api_dir / "tmdb")),
            tvdb=None,
        )

        assert run.still_paths
        assert all(path.exists() for path in run.still_paths.values())
        assert all(
            path.parent == ImageCache(cache.stills_dir).directory
            for path in run.still_paths.values()
        )
