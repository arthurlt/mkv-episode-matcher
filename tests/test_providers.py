"""Tests for the TMDB and TheTVDB clients and the merged season metadata."""

from __future__ import annotations

import httpx
import pytest

from mkv_episode_matcher.cache import JsonCache
from mkv_episode_matcher.metadata import collect_season, describe_still_coverage
from mkv_episode_matcher.models import StillKind
from mkv_episode_matcher.providers import ProviderError, SeriesNotFoundError
from mkv_episode_matcher.tmdb_client import TMDB_IMAGE_BASE, TmdbClient
from mkv_episode_matcher.tvdb_client import TvdbClient


@pytest.fixture
def tmdb(recorded_api, cache_dir):
    return TmdbClient(
        api_key="test-key",
        client=recorded_api.client(),
        cache=JsonCache(cache_dir / "tmdb"),
    )


@pytest.fixture
def tvdb(recorded_api, cache_dir):
    return TvdbClient(
        api_key="test-key",
        client=recorded_api.client(),
        cache=JsonCache(cache_dir / "tvdb"),
    )


class TestTmdbClient:
    def test_resolves_a_series_by_name(self, tmdb):
        assert tmdb.resolve_series("Test Precinct").provider_id == "4224"

    def test_prefers_an_exact_title_over_a_more_popular_near_match(self, tmdb):
        assert tmdb.resolve_series("Test Precinct").name == "Test Precinct"

    def test_can_narrow_the_search_by_year(self, tmdb):
        assert tmdb.resolve_series("Test Precinct", year=1998).name == "Test Precinct: The Reunion"

    def test_unknown_series_raises(self, tmdb):
        with pytest.raises(SeriesNotFoundError):
            tmdb.resolve_series("Nothing Like This Exists")

    def test_punctuation_differences_do_not_defeat_the_match(self, tmdb):
        assert tmdb.resolve_series("test-precinct").provider_id == "4224"

    def test_lists_the_season_episodes(self, tmdb):
        episodes = tmdb.season_episodes("4224", 2)

        assert [episode.number for episode in episodes] == [1, 2, 3, 4]
        assert episodes[0].title == "Chipped Beef"
        assert episodes[0].code == "S02E01"

    def test_collects_every_still_for_an_episode(self, tmdb):
        episodes = tmdb.collect_season("4224", 2)

        by_number = {episode.number: episode for episode in episodes}
        assert len(by_number[1].stills) == 3
        assert len(by_number[2].stills) == 1

    def test_an_episode_with_no_stills_is_reported_empty_not_dropped(self, tmdb):
        by_number = {e.number: e for e in tmdb.collect_season("4224", 2)}

        assert by_number[4].stills == ()
        assert by_number[4].title == "Rites of Spring"

    def test_still_urls_use_the_configured_image_size(self, tmdb):
        episodes = tmdb.collect_season("4224", 2)

        assert episodes[0].stills[0].url.startswith(f"{TMDB_IMAGE_BASE}/w780/")

    def test_stills_are_deduplicated_across_endpoints(self, tmdb):
        """The season list and the images endpoint both mention ``s02e01-a.jpg``."""
        episode = next(e for e in tmdb.collect_season("4224", 2) if e.number == 1)

        urls = [still.url for still in episode.stills]
        assert len(urls) == len(set(urls))

    def test_responses_are_cached_on_disk(self, recorded_api, cache_dir):
        def run() -> None:
            TmdbClient(
                api_key="k",
                client=recorded_api.client(),
                cache=JsonCache(cache_dir / "tmdb"),
            ).collect_season("4224", 2)

        run()
        after_cold_run = len(recorded_api.requests)
        run()

        assert after_cold_run > 0
        assert len(recorded_api.requests) == after_cold_run

    def test_a_v3_key_is_sent_as_a_query_parameter(self, recorded_api, cache_dir):
        TmdbClient(
            api_key="plain-v3-key", client=recorded_api.client(), cache=JsonCache(cache_dir)
        ).season_episodes("4224", 2)

        assert "api_key=plain-v3-key" in str(recorded_api.requests[-1].url)

    def test_a_v4_token_is_sent_as_a_bearer_header(self, recorded_api, cache_dir):
        token = "eyJhbGciOi.eyJhdWQiOi.signature"
        TmdbClient(
            api_key=token, client=recorded_api.client(), cache=JsonCache(cache_dir)
        ).season_episodes("4224", 2)

        request = recorded_api.requests[-1]
        assert request.headers["authorization"] == f"Bearer {token}"
        assert "api_key" not in str(request.url)

    def test_a_missing_season_raises_a_provider_error(self, recorded_api, cache_dir):
        client = TmdbClient(api_key="k", client=recorded_api.client(), cache=JsonCache(cache_dir))

        with pytest.raises(ProviderError):
            client.season_episodes("4224", 99)


class TestTvdbClient:
    def test_logs_in_once_and_reuses_the_token(self, tvdb, recorded_api):
        tvdb.season_episodes("70328", 2)
        tvdb.season_episodes("70328", 2)

        assert recorded_api.paths_called("/login") == 1

    def test_sends_the_bearer_token_on_data_requests(self, tvdb, recorded_api):
        tvdb.season_episodes("70328", 2)

        data_request = next(r for r in recorded_api.requests if "episodes" in str(r.url))
        assert data_request.headers["authorization"] == "Bearer recorded-tvdb-token"

    def test_resolves_a_series_by_name(self, tvdb):
        assert tvdb.resolve_series("Test Precinct").provider_id == "70328"

    def test_lists_the_season_episodes(self, tvdb):
        episodes = tvdb.season_episodes("70328", 2)

        assert [episode.number for episode in episodes] == [1, 2, 3, 4]
        assert episodes[2].title == "The Second Oldest Profession"

    def test_episode_images_become_absolute_artwork_urls(self, tvdb):
        episodes = tvdb.collect_season("70328", 2)

        assert episodes[0].stills[0].url.startswith("https://artworks.thetvdb.com/")

    def test_blank_image_fields_are_ignored(self, tvdb):
        by_number = {e.number: e for e in tvdb.collect_season("70328", 2, extended=False)}

        assert by_number[3].stills == ()

    def test_extended_lookup_adds_screencap_artwork(self, tvdb):
        plain = {e.number: e for e in tvdb.collect_season("70328", 2, extended=False)}
        extended = {e.number: e for e in tvdb.collect_season("70328", 2, extended=True)}

        assert len(extended[1].stills) > len(plain[1].stills)

    def test_extended_lookup_skips_non_screencap_artwork(self, tvdb):
        episode = next(e for e in tvdb.collect_season("70328", 2, extended=True) if e.number == 1)

        assert all(still.kind is not StillKind.PROMOTIONAL for still in episode.stills)
        assert not any("posters" in still.url for still in episode.stills)

    def test_a_pin_is_included_in_the_login_payload(self, recorded_api, cache_dir):
        client = TvdbClient(
            api_key="k", pin="1234", client=recorded_api.client(), cache=JsonCache(cache_dir)
        )
        client.season_episodes("70328", 2)

        login = next(r for r in recorded_api.requests if "login" in str(r.url))
        assert b'"pin"' in login.content

    def test_a_failed_login_raises(self, cache_dir):
        transport = httpx.MockTransport(lambda r: httpx.Response(401, json={"status": "failure"}))
        client = TvdbClient(
            api_key="bad", client=httpx.Client(transport=transport), cache=JsonCache(cache_dir)
        )

        with pytest.raises(ProviderError):
            client.season_episodes("70328", 2)


class TestRetries:
    def test_a_rate_limited_request_is_retried(self, cache_dir, monkeypatch):
        import mkv_episode_matcher.providers as providers

        monkeypatch.setattr(providers, "sleep", lambda _seconds: None)
        responses = [
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(200, json={"results": [{"id": 1, "name": "X"}]}),
        ]
        client = TmdbClient(
            api_key="k",
            client=httpx.Client(transport=httpx.MockTransport(lambda r: responses.pop(0))),
            cache=JsonCache(cache_dir),
        )

        assert client.search_series("X")[0].name == "X"
        assert responses == []

    def test_retries_are_bounded(self, cache_dir, monkeypatch):
        import mkv_episode_matcher.providers as providers

        monkeypatch.setattr(providers, "sleep", lambda _seconds: None)
        calls = []

        def always_busy(request):
            calls.append(request)
            return httpx.Response(503)

        client = TmdbClient(
            api_key="k",
            client=httpx.Client(transport=httpx.MockTransport(always_busy)),
            cache=JsonCache(cache_dir),
            max_retries=2,
        )

        with pytest.raises(ProviderError):
            client.search_series("X")
        assert len(calls) == 3


class TestCollectSeason:
    def test_merges_stills_from_both_providers(self, tmdb, tvdb):
        episodes = collect_season(
            season=2, tmdb=tmdb, tvdb=tvdb, tmdb_series="4224", tvdb_series="70328"
        )

        first = next(e for e in episodes if e.number == 1)
        providers = {still.provider for still in first.stills}
        assert providers == {"tmdb", "tvdb"}

    def test_episode_numbers_are_unique_and_sorted(self, tmdb, tvdb):
        episodes = collect_season(
            season=2, tmdb=tmdb, tvdb=tvdb, tmdb_series="4224", tvdb_series="70328"
        )

        numbers = [episode.number for episode in episodes]
        assert numbers == sorted(numbers) == [1, 2, 3, 4]

    def test_titles_prefer_tmdb(self, tmdb, tvdb):
        episodes = collect_season(
            season=2, tmdb=tmdb, tvdb=tvdb, tmdb_series="4224", tvdb_series="70328"
        )

        assert episodes[0].title == "Chipped Beef"

    def test_records_which_providers_contributed(self, tmdb, tvdb):
        episodes = collect_season(
            season=2, tmdb=tmdb, tvdb=tvdb, tmdb_series="4224", tvdb_series="70328"
        )

        assert set(episodes[0].providers) == {"tmdb", "tvdb"}

    def test_works_with_only_one_provider(self, tmdb):
        episodes = collect_season(season=2, tmdb=tmdb, tvdb=None, tmdb_series="4224")

        assert len(episodes) == 4
        assert all(still.provider == "tmdb" for e in episodes for still in e.stills)

    def test_requires_at_least_one_provider(self):
        with pytest.raises(ValueError, match="provider"):
            collect_season(season=2, tmdb=None, tvdb=None)

    def test_a_failing_provider_does_not_sink_the_run(self, tmdb, cache_dir, caplog):
        broken = TvdbClient(
            api_key="bad",
            client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
            cache=JsonCache(cache_dir / "broken"),
            max_retries=0,
        )

        episodes = collect_season(
            season=2, tmdb=tmdb, tvdb=broken, tmdb_series="4224", tvdb_series="70328"
        )

        assert len(episodes) == 4
        assert "tvdb" in caplog.text.lower()

    def test_coverage_summary_flags_episodes_without_stills(self, tmdb, tvdb):
        episodes = collect_season(
            season=2, tmdb=tmdb, tvdb=tvdb, tmdb_series="4224", tvdb_series="70328"
        )

        coverage = describe_still_coverage(episodes)

        assert coverage.total_episodes == 4
        assert coverage.episodes_without_stills == ()
        assert coverage.total_stills >= 6

    def test_coverage_summary_lists_blind_episodes(self, tmdb):
        episodes = collect_season(season=2, tmdb=tmdb, tvdb=None, tmdb_series="4224")

        assert describe_still_coverage(episodes).episodes_without_stills == ("S02E04",)
