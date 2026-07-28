"""Data types shared across the matching pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

__all__ = [
    "Episode",
    "EpisodeScore",
    "FileScores",
    "MatchResult",
    "MatchStatus",
    "SkipReason",
    "Still",
    "StillHit",
    "StillKind",
    "VideoFile",
    "dedupe_stills",
]


class SkipReason(StrEnum):
    """Why a file was excluded from matching before any frames were hashed."""

    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    UNREADABLE = "unreadable"


class MatchStatus(StrEnum):
    """Outcome of matching one video file against the season's episodes."""

    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"
    SKIPPED = "skipped"


class StillKind(StrEnum):
    """How much a provider image is trusted to be an actual frame grab.

    ``SCREENCAP`` images are literal frames lifted from the episode and match
    best. ``PROMOTIONAL`` images are posed or colour-graded and match worst.
    """

    SCREENCAP = "screencap"
    THUMBNAIL = "thumbnail"
    PROMOTIONAL = "promotional"


@dataclass(frozen=True, slots=True)
class VideoFile:
    """A candidate rip on disk together with the facts used for cache keys.

    Attributes
    ----------
    path
        Location of the ``.mkv`` file.
    duration_s
        Container duration in seconds, from ``ffprobe``.
    size_bytes, mtime_ns
        Used to invalidate a persisted frame-hash index when the file changes.
    """

    path: Path
    duration_s: float
    size_bytes: int
    mtime_ns: int

    @property
    def name(self) -> str:
        """Return the file name without directories."""
        return self.path.name


@dataclass(frozen=True, slots=True)
class Still:
    """One provider image that may appear somewhere inside an episode.

    Examples
    --------
    >>> still = Still(provider="tmdb", url="https://img/abc.jpg", kind=StillKind.SCREENCAP)
    >>> still.filename.endswith(".jpg")
    True
    """

    provider: str
    url: str
    kind: StillKind = StillKind.SCREENCAP
    width: int | None = None
    height: int | None = None

    @property
    def cache_key(self) -> str:
        """Return a stable, filesystem-safe identifier derived from the URL."""
        return hashlib.sha256(self.url.encode()).hexdigest()[:20]

    @property
    def filename(self) -> str:
        """Return the on-disk cache file name for this still."""
        suffix = Path(self.url.split("?")[0]).suffix.lower() or ".jpg"
        return f"{self.provider}-{self.cache_key}{suffix}"


def dedupe_stills(stills: tuple[Still, ...]) -> tuple[Still, ...]:
    """Drop repeated still URLs while preserving order.

    Providers list the same image from more than one endpoint, and hashing a
    duplicate would inflate the multi-still agreement count with no new
    evidence behind it.

    Examples
    --------
    >>> one = Still("tmdb", "https://img/a.jpg")
    >>> len(dedupe_stills((one, one, Still("tmdb", "https://img/b.jpg"))))
    2
    """
    seen: set[str] = set()
    unique: list[Still] = []
    for still in stills:
        if still.url in seen:
            continue
        seen.add(still.url)
        unique.append(still)
    return tuple(unique)


@dataclass(frozen=True, slots=True)
class Episode:
    """A single episode of the requested season, with every still we could find.

    Examples
    --------
    >>> Episode(season=2, number=7, title="Bart Gets an Elephant").code
    'S02E07'
    """

    season: int
    number: int
    title: str
    stills: tuple[Still, ...] = ()
    runtime_minutes: int | None = None
    providers: tuple[str, ...] = ()

    @property
    def code(self) -> str:
        """Return the Plex-style ``SxxEyy`` episode code."""
        return f"S{self.season:02d}E{self.number:02d}"

    def with_stills(self, stills: tuple[Still, ...]) -> Episode:
        """Return a copy of this episode carrying ``stills``."""
        return Episode(
            season=self.season,
            number=self.number,
            title=self.title,
            stills=stills,
            runtime_minutes=self.runtime_minutes,
            providers=self.providers,
        )


@dataclass(frozen=True, slots=True)
class StillHit:
    """The best position of one still inside one file's frame index."""

    still: Still
    distance: int
    timestamp: float
    dhash_distance: int


@dataclass(frozen=True, slots=True)
class EpisodeScore:
    """How well one episode's stills matched one file's frame index.

    Attributes
    ----------
    best_hit
        The single strongest still hit, or ``None`` when the episode has no
        stills at all.
    supporting_stills
        How many distinct stills of this episode landed at or below the hit
        threshold. Multi-still agreement is stronger evidence than one hit.
    cost
        Assignment cost. With pHash-only scoring this is the best Hamming
        distance (discounted for agreement). After NCC verification it is
        ``1 - ncc``. Lower is better.
    ncc
        Best normalized cross-correlation from the verification pass, or
        ``None`` when verification did not run for this pair.
    """

    episode: Episode
    best_hit: StillHit | None
    supporting_stills: int
    cost: float
    ncc: float | None = None

    @property
    def best_distance(self) -> int | None:
        """Return the Hamming distance of the strongest hit, if any."""
        return None if self.best_hit is None else self.best_hit.distance


@dataclass(frozen=True, slots=True)
class FileScores:
    """Every episode score computed for a single video file."""

    video: VideoFile
    scores: tuple[EpisodeScore, ...]

    def ranked(self) -> list[EpisodeScore]:
        """Return the episode scores from best (lowest cost) to worst."""
        return sorted(self.scores, key=lambda score: (score.cost, score.episode.number))


@dataclass(slots=True)
class MatchResult:
    """The final verdict for one file, ready to be reported or renamed."""

    video: VideoFile
    status: MatchStatus
    episode: Episode | None = None
    hit: StillHit | None = None
    cost: float | None = None
    runner_up: Episode | None = None
    runner_up_cost: float | None = None
    supporting_stills: int = 0
    confidence: float | None = None
    ncc: float | None = None
    skip_reason: SkipReason | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def gap(self) -> float | None:
        """Return how much worse the runner-up episode is than the match."""
        if self.cost is None or self.runner_up_cost is None:
            return None
        return self.runner_up_cost - self.cost

    @property
    def renameable(self) -> bool:
        """Return whether this result may be auto-renamed.

        Confident ``matched`` verdicts qualify, including closed-world unique
        runtime fallbacks when provider stills were inconclusive.

        Examples
        --------
        >>> from pathlib import Path
        >>> video = VideoFile(Path("t00.mkv"), 1320.0, 1, 1)
        >>> MatchResult(video, MatchStatus.AMBIGUOUS).renameable
        False
        """
        return self.status is MatchStatus.MATCHED and self.episode is not None
