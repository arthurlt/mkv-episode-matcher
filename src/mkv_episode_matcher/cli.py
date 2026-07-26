"""Command line entry point: scan a folder of rips, identify each episode."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console

from .cache import CacheRoot, JsonCache
from .frames import IndexParams
from .logging_utils import configure_logging
from .metadata import describe_still_coverage
from .models import MatchStatus
from .pipeline import MatchRequest, MatchRun, run_match
from .probe import DEFAULT_MIN_DURATION_S, DurationFilter
from .providers import ProviderError
from .rename import apply_renames, plan_renames
from .report import build_payload, export_previews, render_table, write_json
from .score_visual import ScoringConfig
from .tmdb_client import DEFAULT_IMAGE_SIZE, TmdbClient
from .tvdb_client import TvdbClient

__all__ = ["app", "build_http_client"]

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Identify MakeMKV rips of a known series and season by finding TMDB and "
        "TheTVDB screencaps inside the video itself."
    ),
)

#: Exit code used when files remain ambiguous or unmatched under --strict.
EXIT_UNRESOLVED = 2


def build_http_client() -> httpx.Client:
    """Return the HTTP client used for provider calls and still downloads.

    Exposed as a seam so tests can replay recorded responses through a mock
    transport while the rest of the client stack runs for real.
    """
    return httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": "mkv-episode-matcher/0.1 (+https://github.com)"},
    )


@app.command()
def match(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder of MakeMKV rips (title_t00.mkv and friends).",
        ),
    ],
    series: Annotated[
        str, typer.Option("--series", "-s", help="Series name, e.g. 'Hill Street Blues'.")
    ],
    season: Annotated[int, typer.Option("--season", "-n", min=0, help="Season number.")],
    year: Annotated[
        int | None, typer.Option("--year", help="First-air year, to disambiguate the series.")
    ] = None,
    tmdb_id: Annotated[
        str | None, typer.Option("--tmdb-id", help="Skip the TMDB search and use this series id.")
    ] = None,
    tvdb_id: Annotated[
        str | None,
        typer.Option("--tvdb-id", help="Skip the TheTVDB search and use this series id."),
    ] = None,
    tmdb_key: Annotated[
        str | None, typer.Option(envvar="TMDB_API_KEY", help="TMDB v3 key or v4 token.")
    ] = None,
    tvdb_key: Annotated[
        str | None, typer.Option(envvar="TVDB_API_KEY", help="TheTVDB v4 API key.")
    ] = None,
    tvdb_pin: Annotated[
        str | None, typer.Option(envvar="TVDB_PIN", help="TheTVDB subscriber PIN, if required.")
    ] = None,
    image_size: Annotated[
        str, typer.Option("--image-size", help="TMDB image size to download.")
    ] = DEFAULT_IMAGE_SIZE,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache", envvar="MKV_MATCHER_CACHE", help="Cache root directory."),
    ] = None,
    interval: Annotated[
        float,
        typer.Option("--interval", min=0.05, help="Seconds between sampled frames."),
    ] = 1.0,
    sample_width: Annotated[
        int, typer.Option("--sample-width", min=32, help="Width frames are scaled to for hashing.")
    ] = 320,
    skip_head: Annotated[
        float, typer.Option("--skip-head", min=0.0, help="Seconds to ignore at the start.")
    ] = 0.0,
    skip_tail: Annotated[
        float, typer.Option("--skip-tail", min=0.0, help="Seconds to ignore at the end.")
    ] = 0.0,
    min_minutes: Annotated[
        float,
        typer.Option(
            "--min-minutes", min=0.0, help="Skip files shorter than this (menus, trailers)."
        ),
    ] = DEFAULT_MIN_DURATION_S / 60,
    max_minutes: Annotated[
        float | None,
        typer.Option("--max-minutes", help="Skip files longer than this. Off by default."),
    ] = None,
    threshold: Annotated[
        int,
        typer.Option("--threshold", "-t", min=0, max=64, help="Max pHash distance for a hit."),
    ] = 12,
    gap: Annotated[
        float,
        typer.Option("--gap", "-g", min=0.0, help="Margin the runner-up must lose by."),
    ] = 4.0,
    workers: Annotated[
        int, typer.Option("--workers", "-w", min=1, help="Concurrent ffmpeg and download workers.")
    ] = 4,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Rebuild frame indexes even if cached.")
    ] = False,
    refine: Annotated[
        bool,
        typer.Option("--refine", help="Two-stage mode: re-sample densely around promising hits."),
    ] = False,
    refine_interval: Annotated[
        float, typer.Option("--refine-interval", min=0.01, help="Interval for the refine pass.")
    ] = 0.25,
    no_tmdb: Annotated[bool, typer.Option("--no-tmdb", help="Do not query TMDB.")] = False,
    no_tvdb: Annotated[bool, typer.Option("--no-tvdb", help="Do not query TheTVDB.")] = False,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the full report as JSON to this path.")
    ] = None,
    previews: Annotated[
        Path | None,
        typer.Option("--previews", help="Write side-by-side still/frame previews to this folder."),
    ] = None,
    rename: Annotated[
        bool, typer.Option("--rename", help="Rename confidently matched files (Plex layout).")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Allow a rename to replace an existing file.")
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit non-zero if any file is unresolved.")
    ] = False,
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="Increase logging (-v, -vv).")
    ] = 0,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only report errors.")] = False,
) -> None:
    """Match every rip in INPUT_DIR to an episode of the given series and season."""
    configure_logging(verbose, quiet=quiet)
    console = Console()

    if no_tmdb and no_tvdb:
        raise typer.BadParameter("--no-tmdb and --no-tvdb cannot both be given")

    cache = CacheRoot(cache_dir) if cache_dir else CacheRoot.default()
    logger.info("using cache at %s", cache.path)

    http = build_http_client()
    try:
        tmdb = None
        if not no_tmdb:
            if not tmdb_key:
                raise typer.BadParameter("TMDB_API_KEY is not set; pass --tmdb-key or --no-tmdb")
            tmdb = TmdbClient(
                api_key=tmdb_key,
                client=http,
                cache=JsonCache(cache.api_dir / "tmdb"),
                image_size=image_size,
                max_workers=workers,
            )

        tvdb = None
        if not no_tvdb:
            if not tvdb_key:
                raise typer.BadParameter("TVDB_API_KEY is not set; pass --tvdb-key or --no-tvdb")
            tvdb = TvdbClient(
                api_key=tvdb_key,
                pin=tvdb_pin,
                client=http,
                cache=JsonCache(cache.api_dir / "tvdb"),
                max_workers=workers,
            )

        request = MatchRequest(
            input_dir=input_dir,
            series=series,
            season=season,
            year=year,
            tmdb_series_id=tmdb_id,
            tvdb_series_id=tvdb_id,
            index_params=IndexParams(
                interval_s=interval,
                skip_head_s=skip_head,
                skip_tail_s=skip_tail,
                sample_width=sample_width,
            ),
            scoring=ScoringConfig(match_threshold=threshold, decision_gap=gap),
            duration_filter=DurationFilter(
                min_duration_s=min_minutes * 60,
                max_duration_s=None if max_minutes is None else max_minutes * 60,
            ),
            max_workers=workers,
            refresh_index=refresh,
            refine=refine,
            refine_interval_s=refine_interval,
        )

        try:
            run = run_match(request, http=http, cache=cache, tmdb=tmdb, tvdb=tvdb)
        except ProviderError as error:
            logger.error("%s", error)
            raise typer.Exit(code=1) from error
    finally:
        http.close()

    _emit(run, console=console, series=series, season=season, json_out=json_out, previews=previews)

    if rename:
        _rename(run, console=console, series=series, force=force)

    if strict and run.unresolved:
        console.print(f"[yellow]{len(run.unresolved)} file(s) still need a human.[/yellow]")
        raise typer.Exit(code=EXIT_UNRESOLVED)


def _emit(
    run: MatchRun,
    *,
    console: Console,
    series: str,
    season: int,
    json_out: Path | None,
    previews: Path | None,
) -> None:
    """Print the table and write any requested artefacts."""
    console.print(render_table(run.results))
    console.print(describe_still_coverage(run.episodes).summary())

    counts = {status.value: 0 for status in MatchStatus}
    for result in run.results:
        counts[result.status.value] += 1
    console.print(
        " ".join(f"{name}={count}" for name, count in counts.items()) + f" in {run.elapsed_s:.1f}s"
    )

    if json_out is not None:
        write_json(json_out, build_payload(run.results, series=series, season=season))
        console.print(f"JSON report: {json_out}")

    if previews is not None:
        written = export_previews(run.results, still_paths=run.still_paths, destination=previews)
        console.print(f"Wrote {len(written)} preview image(s) to {previews}")


def _rename(run: MatchRun, *, console: Console, series: str, force: bool) -> None:
    """Apply renames for confidently matched files only."""
    plans = plan_renames(run.results, series=series)
    if not plans:
        console.print("Nothing to rename: no file reached a confident match.")
        return
    for outcome in apply_renames(plans, dry_run=False, force=force):
        style = "green" if outcome.applied else "yellow"
        console.print(f"[{style}]{outcome.description}[/{style}]")


@app.command("clear-cache")
def clear_cache(
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache", envvar="MKV_MATCHER_CACHE", help="Cache root directory."),
    ] = None,
    frames_only: Annotated[
        bool, typer.Option("--frames-only", help="Only drop the frame-hash indexes.")
    ] = False,
) -> None:
    """Delete cached API responses, stills, and frame indexes."""
    configure_logging(0)
    cache = CacheRoot(cache_dir) if cache_dir else CacheRoot.default()
    targets = [cache.frames_dir] if frames_only else [cache.path]
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
    typer.echo(f"cleared {', '.join(str(target) for target in targets)}")


if __name__ == "__main__":  # pragma: no cover - module entry point
    app()
