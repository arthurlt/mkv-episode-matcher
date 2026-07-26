"""TheTVDB v4 client, biased towards genuine episode screencaps.

TheTVDB is worth querying alongside TMDB precisely because its episode artwork
is usually a real frame grab rather than a promotional photo, and a second
independent set of frames materially raises the chance that at least one still
lands somewhere the matcher can find it.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import httpx

from .cache import JsonCache
from .models import Episode, Still, StillKind, dedupe_stills
from .providers import JsonApiClient, ProviderError, SeriesMatch, choose_series

__all__ = ["TVDB_API_BASE", "TVDB_ARTWORK_BASE", "TvdbClient"]

logger = logging.getLogger(__name__)

TVDB_API_BASE = "https://api4.thetvdb.com/v4"
TVDB_ARTWORK_BASE = "https://artworks.thetvdb.com"

#: TheTVDB artwork type ids that denote an actual frame from the episode.
SCREENCAP_ARTWORK_TYPES = frozenset({12})

_MAX_PAGES = 20


class TvdbClient(JsonApiClient):
    """Read-only TheTVDB access for one series and season."""

    provider = "tvdb"
    base_url = TVDB_API_BASE

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.Client,
        cache: JsonCache,
        pin: str | None = None,
        season_type: str = "default",
        max_retries: int = 3,
        max_workers: int = 4,
    ) -> None:
        super().__init__(client=client, cache=cache, max_retries=max_retries)
        self.api_key = api_key
        self.pin = pin
        self.season_type = season_type
        self.max_workers = max_workers
        self._token: str | None = None

    def token(self) -> str:
        """Return a bearer token, logging in once per client instance."""
        if self._token is None:
            body: dict[str, object] = {"apikey": self.api_key}
            if self.pin:
                body["pin"] = self.pin
            payload = self.request_json("POST", "/login", json_body=body)
            token = payload.get("data", {}).get("token")
            if not token:
                raise ProviderError("thetvdb login returned no token")
            self._token = str(token)
            logger.debug("authenticated against thetvdb")
        return self._token

    def _get(self, path: str, *, extra: dict[str, object] | None = None, cache_key: str) -> dict:
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        payload = self.get_json(
            path,
            params=extra,
            headers={"Authorization": f"Bearer {self.token()}"},
            cache_key=cache_key,
        )
        return payload

    @staticmethod
    def artwork_url(image: str) -> str:
        """Turn a TheTVDB image path into an absolute artwork URL.

        Examples
        --------
        >>> TvdbClient.artwork_url("/banners/episodes/1/2.jpg")
        'https://artworks.thetvdb.com/banners/episodes/1/2.jpg'
        >>> TvdbClient.artwork_url("https://example.test/a.jpg")
        'https://example.test/a.jpg'
        """
        if image.startswith("http"):
            return image
        return f"{TVDB_ARTWORK_BASE}/{image.lstrip('/')}"

    def search_series(self, name: str) -> list[SeriesMatch]:
        """Return the series TheTVDB knows about under ``name``."""
        payload = self._get(
            "/search",
            extra={"query": name, "type": "series"},
            cache_key=f"tvdb/search/{name}",
        )
        matches = []
        for result in payload.get("data", []):
            year = str(result.get("year") or "")
            matches.append(
                SeriesMatch(
                    provider="tvdb",
                    provider_id=str(result.get("tvdb_id") or result.get("id")),
                    name=str(result.get("name") or ""),
                    year=int(year) if year.isdigit() else None,
                )
            )
        return matches

    def resolve_series(self, name: str, year: int | None = None) -> SeriesMatch:
        """Resolve ``name`` to a single TheTVDB series."""
        match = choose_series("tvdb", self.search_series(name), name, year)
        logger.info("resolved series on thetvdb: %s", match)
        return match

    def season_episodes(self, series_id: str, season: int) -> list[Episode]:
        """Return the season's episodes with their primary episode image."""
        episodes: list[Episode] = []
        for page in range(_MAX_PAGES):
            payload = self._get(
                f"/series/{series_id}/episodes/{self.season_type}",
                extra={"season": season, "page": page},
                cache_key=f"tvdb/episodes/{series_id}/{self.season_type}/{season}/{page}",
            )
            data = payload.get("data") or {}
            raw_episodes = data.get("episodes", [])
            for raw in raw_episodes:
                if int(raw.get("seasonNumber", season)) != season:
                    continue
                image = str(raw.get("image") or "")
                episodes.append(
                    Episode(
                        season=season,
                        number=int(raw["number"]),
                        title=str(raw.get("name") or ""),
                        stills=(
                            (
                                Still(
                                    provider="tvdb",
                                    url=self.artwork_url(image),
                                    kind=StillKind.SCREENCAP,
                                ),
                            )
                            if image
                            else ()
                        ),
                        runtime_minutes=raw.get("runtime"),
                        providers=("tvdb",),
                    )
                )
            if not (payload.get("links") or {}).get("next"):
                break

        return sorted(episodes, key=lambda episode: episode.number)

    def episode_ids(self, series_id: str, season: int) -> dict[int, int]:
        """Map episode number to TheTVDB episode id for the given season."""
        payload = self._get(
            f"/series/{series_id}/episodes/{self.season_type}",
            extra={"season": season, "page": 0},
            cache_key=f"tvdb/episodes/{series_id}/{self.season_type}/{season}/0",
        )
        data = payload.get("data") or {}
        return {
            int(raw["number"]): int(raw["id"])
            for raw in data.get("episodes", [])
            if int(raw.get("seasonNumber", season)) == season
        }

    def episode_artworks(self, episode_id: int) -> tuple[Still, ...]:
        """Return the screencap artworks TheTVDB holds for one episode."""
        try:
            payload = self._get(
                f"/episodes/{episode_id}/extended",
                cache_key=f"tvdb/episode-extended/{episode_id}",
            )
        except ProviderError as error:
            logger.warning("no extended tvdb record for episode %s (%s)", episode_id, error)
            return ()

        data = payload.get("data") or {}
        stills: list[Still] = []
        primary = str(data.get("image") or "")
        if primary:
            stills.append(
                Still(provider="tvdb", url=self.artwork_url(primary), kind=StillKind.SCREENCAP)
            )
        for artwork in data.get("artworks", []):
            image = str(artwork.get("image") or "")
            if not image:
                continue
            kind = classify_artwork(artwork.get("type"), image)
            if kind is StillKind.PROMOTIONAL:
                continue
            stills.append(
                Still(
                    provider="tvdb",
                    url=self.artwork_url(image),
                    kind=kind,
                    width=artwork.get("width"),
                    height=artwork.get("height"),
                )
            )
        return tuple(stills)

    def collect_season(
        self, series_id: str, season: int, *, extended: bool = True
    ) -> list[Episode]:
        """Return the season's episodes with all usable TheTVDB screencaps.

        Parameters
        ----------
        series_id
            TheTVDB series id.
        season
            Season number.
        extended
            Also fetch each episode's extended record, which often carries
            additional frame grabs beyond the single headline image.
        """
        episodes = self.season_episodes(series_id, season)
        if not episodes or not extended:
            return episodes

        ids = self.episode_ids(series_id, season)
        with ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, len(episodes)))) as pool:
            extra = list(
                pool.map(
                    lambda episode: self.episode_artworks(ids[episode.number])
                    if episode.number in ids
                    else (),
                    episodes,
                )
            )

        merged = []
        for episode, stills in zip(episodes, extra, strict=True):
            merged.append(episode.with_stills(dedupe_stills(episode.stills + stills)))
        return merged


def classify_artwork(type_id: object, image: str) -> StillKind:
    """Classify a TheTVDB artwork entry by how frame-like it is.

    Examples
    --------
    >>> classify_artwork(12, "/banners/episodes/1/2.jpg").value
    'screencap'
    >>> classify_artwork(2, "/banners/series/1/posters/a.jpg").value
    'promotional'
    """
    try:
        numeric = int(type_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        numeric = -1
    if numeric in SCREENCAP_ARTWORK_TYPES:
        return StillKind.SCREENCAP
    if "/episodes/" in image:
        return StillKind.SCREENCAP
    return StillKind.PROMOTIONAL

