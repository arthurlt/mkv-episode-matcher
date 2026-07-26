"""TMDB client: season episode lists and *every* still per episode.

The season endpoint only exposes one "primary" still per episode. The images
endpoint exposes the rest, and those extra frames are exactly the evidence this
tool runs on, so both are collected and merged.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import httpx

from .cache import JsonCache
from .models import Episode, Still, StillKind, dedupe_stills
from .providers import JsonApiClient, ProviderError, SeriesMatch, choose_series

__all__ = ["TMDB_API_BASE", "TMDB_IMAGE_BASE", "TmdbClient"]

logger = logging.getLogger(__name__)

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"

#: Wide enough to preserve composition, small enough to download in bulk.
DEFAULT_IMAGE_SIZE = "w780"


class TmdbClient(JsonApiClient):
    """Read-only TMDB access for one series and season."""

    provider = "tmdb"
    base_url = TMDB_API_BASE

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.Client,
        cache: JsonCache,
        image_size: str = DEFAULT_IMAGE_SIZE,
        language: str = "en-US",
        max_retries: int = 3,
        max_workers: int = 4,
    ) -> None:
        super().__init__(client=client, cache=cache, max_retries=max_retries)
        self.api_key = api_key
        self.image_size = image_size
        self.language = language
        self.max_workers = max_workers

    @property
    def uses_bearer_token(self) -> bool:
        """Return whether the credential is a v4 token rather than a v3 key.

        Examples
        --------
        >>> import httpx
        >>> from mkv_episode_matcher.cache import JsonCache
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     client = TmdbClient(
        ...         api_key="abcdef", client=httpx.Client(), cache=JsonCache(Path(directory))
        ...     )
        ...     client.uses_bearer_token
        False
        """
        return self.api_key.count(".") == 2

    def _auth(self) -> tuple[dict[str, object], dict[str, str]]:
        """Return the ``(params, headers)`` carrying credentials."""
        params: dict[str, object] = {"language": self.language}
        headers: dict[str, str] = {}
        if self.uses_bearer_token:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            params["api_key"] = self.api_key
        return params, headers

    def _get(self, path: str, *, extra: dict[str, object] | None = None, cache_key: str) -> dict:
        params, headers = self._auth()
        params.update(extra or {})
        return self.get_json(path, params=params, headers=headers, cache_key=cache_key)

    def image_url(self, file_path: str) -> str:
        """Turn a TMDB ``file_path`` into a fully qualified CDN URL.

        Examples
        --------
        >>> import httpx, tempfile
        >>> from pathlib import Path
        >>> from mkv_episode_matcher.cache import JsonCache
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     client = TmdbClient(
        ...         api_key="k", client=httpx.Client(), cache=JsonCache(Path(directory))
        ...     )
        ...     client.image_url("/abc.jpg")
        'https://image.tmdb.org/t/p/w780/abc.jpg'
        """
        if file_path.startswith("http"):
            return file_path
        return f"{TMDB_IMAGE_BASE}/{self.image_size}/{file_path.lstrip('/')}"

    def search_series(self, name: str) -> list[SeriesMatch]:
        """Return the series TMDB knows about under ``name``."""
        payload = self._get(
            "/search/tv", extra={"query": name}, cache_key=f"tmdb/search/{name}"
        )
        matches = []
        for result in payload.get("results", []):
            aired = str(result.get("first_air_date") or "")
            matches.append(
                SeriesMatch(
                    provider="tmdb",
                    provider_id=str(result["id"]),
                    name=str(result.get("name") or result.get("original_name") or ""),
                    year=int(aired[:4]) if aired[:4].isdigit() else None,
                )
            )
        return matches

    def resolve_series(self, name: str, year: int | None = None) -> SeriesMatch:
        """Resolve ``name`` to a single TMDB series."""
        match = choose_series("tmdb", self.search_series(name), name, year)
        logger.info("resolved series on tmdb: %s", match)
        return match

    def season_episodes(self, series_id: str, season: int) -> list[Episode]:
        """Return the season's episodes, each carrying its primary still."""
        payload = self._get(
            f"/tv/{series_id}/season/{season}",
            cache_key=f"tmdb/season/{series_id}/{season}",
        )
        episodes = []
        for raw in payload.get("episodes", []):
            still_path = raw.get("still_path")
            stills = (
                (Still(provider="tmdb", url=self.image_url(still_path), kind=StillKind.SCREENCAP),)
                if still_path
                else ()
            )
            episodes.append(
                Episode(
                    season=int(raw.get("season_number", season)),
                    number=int(raw["episode_number"]),
                    title=str(raw.get("name") or ""),
                    stills=stills,
                    runtime_minutes=raw.get("runtime"),
                    providers=("tmdb",),
                )
            )
        return sorted(episodes, key=lambda episode: episode.number)

    def episode_stills(self, series_id: str, season: int, number: int) -> tuple[Still, ...]:
        """Return every still TMDB holds for one episode."""
        try:
            payload = self._get(
                f"/tv/{series_id}/season/{season}/episode/{number}/images",
                cache_key=f"tmdb/images/{series_id}/{season}/{number}",
            )
        except ProviderError as error:
            logger.warning("no tmdb images for S%02dE%02d (%s)", season, number, error)
            return ()

        return tuple(
            Still(
                provider="tmdb",
                url=self.image_url(str(entry["file_path"])),
                kind=StillKind.SCREENCAP,
                width=entry.get("width"),
                height=entry.get("height"),
            )
            for entry in payload.get("stills", [])
            if entry.get("file_path")
        )

    def collect_season(self, series_id: str, season: int) -> list[Episode]:
        """Return the season's episodes with every still TMDB can offer.

        Episodes with no stills are kept in the list. They simply carry no
        visual evidence, which the matcher reports as unmatched rather than
        papering over with a runtime guess.
        """
        episodes = self.season_episodes(series_id, season)
        if not episodes:
            return []

        with ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, len(episodes)))) as pool:
            extra = list(
                pool.map(
                    lambda episode: self.episode_stills(series_id, season, episode.number),
                    episodes,
                )
            )

        merged = []
        for episode, stills in zip(episodes, extra, strict=True):
            merged.append(episode.with_stills(dedupe_stills(episode.stills + stills)))
        return merged

