"""Identify MakeMKV rips by finding provider screencaps inside the video.

Given a folder of rips from a known series and season, the matcher builds a
dense perceptual-hash index of every candidate file, searches each TMDB and
TheTVDB episode still against those indexes, and assigns files to episodes on
visual evidence alone. Runtime is used only to discard menus and trailers, and
disc title order is never used as a prior.
"""

from .models import Episode, MatchResult, MatchStatus, SkipReason, Still, VideoFile

__all__ = [
    "Episode",
    "MatchResult",
    "MatchStatus",
    "SkipReason",
    "Still",
    "VideoFile",
    "__version__",
]

__version__ = "0.1.0"
