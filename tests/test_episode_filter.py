"""Tests for episode-number filters used on multi-episode discs."""

from __future__ import annotations

import pytest

from mkv_episode_matcher.episode_filter import filter_episodes_by_number, parse_episode_numbers
from mkv_episode_matcher.models import Episode


class TestParseEpisodeNumbers:
    def test_parses_inclusive_range(self):
        assert parse_episode_numbers("1-7") == frozenset(range(1, 8))

    def test_parses_mixed_list(self):
        assert parse_episode_numbers("1, 3-5, 7") == frozenset({1, 3, 4, 5, 7})

    def test_rejects_empty_range(self):
        with pytest.raises(ValueError, match="at least one"):
            parse_episode_numbers(" , ")

    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="invalid episode range"):
            parse_episode_numbers("7-1")


class TestFilterEpisodesByNumber:
    def test_keeps_only_requested_numbers(self):
        episodes = [
            Episode(season=2, number=1, title="A"),
            Episode(season=2, number=2, title="B"),
            Episode(season=2, number=3, title="C"),
        ]
        filtered = filter_episodes_by_number(episodes, frozenset({1, 3}))
        assert [episode.number for episode in filtered] == [1, 3]

    def test_errors_when_metadata_missing(self):
        episodes = [Episode(season=2, number=1, title="A")]
        with pytest.raises(ValueError, match="E02"):
            filter_episodes_by_number(episodes, frozenset({1, 2}))
