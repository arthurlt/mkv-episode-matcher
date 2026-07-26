"""End-to-end orchestration: scan, index, search, assign.

Keeping the pipeline separate from the command line means the whole matching
run can be exercised in tests against real videos and replayed API responses,
without a terminal in the loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx

from .assign import assign
from .cache import CacheRoot, ImageCache, JsonCache
from .frames import FrameIndexCache, FrameIndexer, IndexParams, build_indexes
from .metadata import StillCoverage, collect_season, describe_still_coverage
from .models import Episode, FileScores, MatchResult, MatchStatus
from .normalize import ImageHashes
from .probe import DurationFilter, find_mkv_files, inventory
from .score_visual import ScoringConfig, hash_stills, score_all, score_episode
from .tmdb_client import TmdbClient
from .tvdb_client import TvdbClient

__all__ = ["MatchRequest", "MatchRun", "run_match"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MatchRequest:
    """Everything one matching run needs to know.

    Attributes
    ----------
    input_dir
        Folder of MakeMKV rips to identify.
    series, season
        The known series and season. v1 never tries to work these out.
    refine
        Enable the two-stage mode: index coarsely, then re-sample densely
        around promising hits before deciding.
    """

    input_dir: Path
    series: str
    season: int
    year: int | None = None
    tmdb_series_id: str | None = None
    tvdb_series_id: str | None = None
    index_params: IndexParams = field(default_factory=IndexParams)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    duration_filter: DurationFilter = field(default_factory=DurationFilter)
    max_workers: int = 4
    refresh_index: bool = False
    tvdb_extended: bool = True
    refine: bool = False
    refine_interval_s: float = 0.25
    refine_window_s: float = 3.0


@dataclass(slots=True)
class MatchRun:
    """The outcome of a run, including the material needed to review it."""

    results: list[MatchResult]
    episodes: list[Episode]
    still_paths: dict[str, Path]
    coverage: StillCoverage
    elapsed_s: float

    @property
    def unresolved(self) -> list[MatchResult]:
        """Return the results a human still has to deal with."""
        return [
            result
            for result in self.results
            if result.status in {MatchStatus.AMBIGUOUS, MatchStatus.UNMATCHED}
        ]


def run_match(
    request: MatchRequest,
    *,
    http: httpx.Client,
    cache: CacheRoot,
    tmdb: TmdbClient | None,
    tvdb: TvdbClient | None,
) -> MatchRun:
    """Run the full pipeline for one folder of rips.

    Parameters
    ----------
    request
        The run configuration.
    http
        HTTP client used for downloading stills.
    cache
        Root of the on-disk caches.
    tmdb, tvdb
        Configured provider clients; at least one is required.

    Returns
    -------
    MatchRun
        Final verdicts plus the episode metadata and local still paths, so the
        caller can render a report or export previews.
    """
    started = time.monotonic()

    paths = find_mkv_files(request.input_dir)
    logger.info("found %d mkv files in %s", len(paths), request.input_dir)
    videos = inventory(paths, on_error="zero")
    candidates, skipped = request.duration_filter.partition(videos)
    logger.info("%d candidates after the duration floor, %d skipped", len(candidates), len(skipped))

    episodes = collect_season(
        season=request.season,
        tmdb=tmdb,
        tvdb=tvdb,
        tmdb_series=request.tmdb_series_id,
        tvdb_series=request.tvdb_series_id,
        series_name=request.series,
        year=request.year,
        tvdb_extended=request.tvdb_extended,
    )
    coverage = describe_still_coverage(episodes)

    images = ImageCache(cache.stills_dir)
    still_hashes = hash_stills(
        [still for episode in episodes for still in episode.stills],
        client=http,
        images=images,
        hash_cache=JsonCache(cache.path / "still-hashes"),
        max_workers=max(4, request.max_workers),
    )
    still_paths = {
        url: images.path_for(still)
        for episode in episodes
        for still in episode.stills
        if (url := still.url) in still_hashes and images.path_for(still).exists()
    }

    indexer = FrameIndexer(FrameIndexCache(cache.frames_dir), request.index_params)
    indexes = build_indexes(
        indexer, candidates, max_workers=request.max_workers, refresh=request.refresh_index
    )
    unindexed = [video for video in candidates if video.path not in indexes]

    scored = score_all(candidates, indexes, episodes, still_hashes, request.scoring)
    if request.refine:
        scored = [_refine(file_scores, indexer, still_hashes, request) for file_scores in scored]

    results = assign(scored, config=request.scoring, skipped=skipped, unindexed=unindexed)
    elapsed = time.monotonic() - started
    logger.info("matched %d files in %.1fs", len(results), elapsed)

    return MatchRun(
        results=results,
        episodes=episodes,
        still_paths=still_paths,
        coverage=coverage,
        elapsed_s=elapsed,
    )


def _refine(
    file_scores: FileScores,
    indexer: FrameIndexer,
    still_hashes: dict[str, ImageHashes],
    request: MatchRequest,
) -> FileScores:
    """Re-sample densely around the leading candidates and re-score them.

    The coarse pass finds roughly where a still lives; a still sitting between
    two coarse samples can look much worse than it is. Re-decoding a few
    seconds around each promising hit costs little and recovers those.
    """
    leaders = [
        (score, score.best_hit) for score in file_scores.ranked()[:2] if score.best_hit is not None
    ]
    if not leaders:
        return file_scores

    video = file_scores.video
    improved = {}
    for score, hit in leaders:
        window = indexer.extract_window(
            video,
            start_s=hit.timestamp - request.refine_window_s,
            end_s=hit.timestamp + request.refine_window_s,
            interval_s=request.refine_interval_s,
        )
        if len(window) == 0:
            continue
        sharpened = score_episode(window, score.episode, still_hashes, request.scoring)
        if sharpened.cost < score.cost:
            logger.debug(
                "refined %s/%s: %.1f -> %.1f",
                video.name,
                score.episode.code,
                score.cost,
                sharpened.cost,
            )
            improved[score.episode.number] = sharpened

    if not improved:
        return file_scores
    return replace(
        file_scores,
        scores=tuple(improved.get(score.episode.number, score) for score in file_scores.scores),
    )
