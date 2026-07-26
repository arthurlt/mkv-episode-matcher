"""Merge one season's episodes and stills from every configured provider.

Providers disagree, go down, and hold different pictures of the same scene.
None of that should stop a run: whatever stills can be gathered are pooled per
episode number, and an episode nobody photographed is reported as having no
visual evidence rather than being quietly guessed at.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import Episode, Still, dedupe_stills
from .providers import ProviderError
from .tmdb_client import TmdbClient
from .tvdb_client import TvdbClient

__all__ = ["StillCoverage", "collect_season", "describe_still_coverage"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StillCoverage:
    """How much visual evidence a season's metadata actually provides."""

    total_episodes: int
    total_stills: int
    episodes_without_stills: tuple[str, ...]

    def summary(self) -> str:
        """Return a one-line human summary of the coverage.

        Examples
        --------
        >>> StillCoverage(4, 9, ("S02E04",)).summary()
        '9 stills across 4 episodes; no stills for S02E04'
        >>> StillCoverage(4, 9, ()).summary()
        '9 stills across 4 episodes'
        """
        base = f"{self.total_stills} stills across {self.total_episodes} episodes"
        if not self.episodes_without_stills:
            return base
        return f"{base}; no stills for {', '.join(self.episodes_without_stills)}"


def describe_still_coverage(episodes: list[Episode]) -> StillCoverage:
    """Summarise how many stills were gathered, and for which episodes none were."""
    return StillCoverage(
        total_episodes=len(episodes),
        total_stills=sum(len(episode.stills) for episode in episodes),
        episodes_without_stills=tuple(episode.code for episode in episodes if not episode.stills),
    )


def collect_season(
    *,
    season: int,
    tmdb: TmdbClient | None = None,
    tvdb: TvdbClient | None = None,
    tmdb_series: str | None = None,
    tvdb_series: str | None = None,
    series_name: str | None = None,
    year: int | None = None,
    tvdb_extended: bool = True,
) -> list[Episode]:
    """Gather the season's episodes and pooled stills from all providers.

    Parameters
    ----------
    season
        Season number to fetch.
    tmdb, tvdb
        Configured provider clients. At least one is required.
    tmdb_series, tvdb_series
        Known provider series ids. When omitted, ``series_name`` is resolved on
        that provider instead.
    series_name, year
        Used to resolve series ids that were not supplied directly.
    tvdb_extended
        Fetch TheTVDB's extended episode records for additional screencaps.

    Returns
    -------
    list of Episode
        One entry per episode number seen on any provider, sorted by number.

    Raises
    ------
    ValueError
        If no provider was supplied.
    """
    if tmdb is None and tvdb is None:
        raise ValueError("at least one metadata provider is required")

    collected: list[list[Episode]] = []
    if tmdb is not None:
        collected.append(
            _safe_collect(
                "tmdb",
                lambda: tmdb.collect_season(
                    tmdb_series
                    or tmdb.resolve_series(_require_name(series_name), year).provider_id,
                    season,
                ),
            )
        )
    if tvdb is not None:
        collected.append(
            _safe_collect(
                "tvdb",
                lambda: tvdb.collect_season(
                    tvdb_series
                    or tvdb.resolve_series(_require_name(series_name), year).provider_id,
                    season,
                    extended=tvdb_extended,
                ),
            )
        )

    merged = _merge(collected)
    logger.info("season %d metadata: %s", season, describe_still_coverage(merged).summary())
    return merged


def _require_name(series_name: str | None) -> str:
    """Return ``series_name`` or explain that it is needed to resolve an id."""
    if not series_name:
        raise ValueError("a series name is required when no provider series id is given")
    return series_name


def _safe_collect(provider: str, fetch) -> list[Episode]:  # noqa: ANN001 - callable returning episodes
    """Run ``fetch``, downgrading a provider outage to a warning."""
    try:
        return fetch()
    except (ProviderError, ValueError) as error:
        logger.error("provider %s is unavailable, continuing without it: %s", provider, error)
        return []


def _merge(sources: list[list[Episode]]) -> list[Episode]:
    """Pool episodes from several providers by episode number."""
    by_number: dict[int, Episode] = {}
    stills_by_number: dict[int, tuple[Still, ...]] = {}
    providers_by_number: dict[int, tuple[str, ...]] = {}

    for episodes in sources:
        for episode in episodes:
            number = episode.number
            existing = by_number.get(number)
            if existing is None or (not existing.title and episode.title):
                by_number[number] = episode
            stills_by_number[number] = stills_by_number.get(number, ()) + episode.stills
            providers_by_number[number] = providers_by_number.get(number, ()) + episode.providers

    merged = []
    for number in sorted(by_number):
        base = by_number[number]
        merged.append(
            Episode(
                season=base.season,
                number=number,
                title=base.title,
                stills=dedupe_stills(stills_by_number[number]),
                runtime_minutes=base.runtime_minutes,
                providers=tuple(dict.fromkeys(providers_by_number[number])),
            )
        )
    return merged
