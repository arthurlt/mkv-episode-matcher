"""Human-readable and machine-readable reporting of match results.

Every matched file is reported with the still that identified it and the
timestamp where that still was found, so verifying a match means jumping to one
second of video rather than scrubbing a whole episode. The optional side-by-side
preview export makes that check purely visual.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from PIL import Image
from rich.table import Table

from .frames import grab_frame
from .models import MatchResult, MatchStatus

__all__ = [
    "build_payload",
    "export_previews",
    "format_timestamp",
    "render_table",
    "table_rows",
    "write_json",
]

logger = logging.getLogger(__name__)

_STATUS_STYLE = {
    MatchStatus.MATCHED: "green",
    MatchStatus.AMBIGUOUS: "yellow",
    MatchStatus.UNMATCHED: "red",
    MatchStatus.SKIPPED: "dim",
}


def format_timestamp(seconds: float | None) -> str:
    """Format a timestamp as ``HH:MM:SS``.

    Examples
    --------
    >>> format_timestamp(612.5)
    '00:10:12'
    >>> format_timestamp(None)
    '-'
    """
    if seconds is None:
        return "-"
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def build_payload(results: Sequence[MatchResult], *, series: str, season: int) -> dict:
    """Assemble the JSON report for a run.

    Parameters
    ----------
    results
        Final verdicts, one per file.
    series, season
        What was being matched, echoed back for provenance.

    Returns
    -------
    dict
        A JSON-serialisable report with a per-file list and a status summary.
    """
    files = []
    for result in results:
        hit = result.hit
        files.append(
            {
                "file": result.video.name,
                "path": str(result.video.path),
                "duration_s": round(result.video.duration_s, 3) or None,
                "status": result.status.value,
                "confidence": result.confidence,
                "episode": result.episode.code if result.episode else None,
                "title": result.episode.title if result.episode else None,
                "distance": hit.distance if hit else None,
                "dhash_distance": hit.dhash_distance if hit else None,
                "supporting_stills": result.supporting_stills,
                "matched_timestamp": round(hit.timestamp, 3) if hit else None,
                "matched_timestamp_hms": format_timestamp(hit.timestamp if hit else None),
                "still_url": hit.still.url if hit else None,
                "still_provider": hit.still.provider if hit else None,
                "runner_up": result.runner_up.code if result.runner_up else None,
                "runner_up_cost": result.runner_up_cost,
                "gap": result.gap,
                "skip_reason": result.skip_reason.value if result.skip_reason else None,
                "notes": list(result.notes),
            }
        )

    summary = {status.value: 0 for status in MatchStatus}
    for result in results:
        summary[result.status.value] += 1

    return {"series": series, "season": season, "summary": summary, "files": files}


def write_json(path: Path, payload: dict) -> Path:
    """Write ``payload`` to ``path`` as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    logger.info("wrote JSON report to %s", path)
    return path


#: Column headings of the console report, in order.
TABLE_COLUMNS = (
    "File",
    "Status",
    "Conf",
    "Episode",
    "Title",
    "Dist",
    "Stills",
    "Found at",
    "Runner-up",
    "Notes",
)


def table_rows(results: Sequence[MatchResult]) -> list[tuple[str, ...]]:
    """Return the report as plain text rows, one per file.

    Kept separate from :func:`render_table` so the content of the report can be
    asserted on without depending on how a terminal happens to wrap it.
    """
    rows = []
    for result in results:
        hit = result.hit
        rows.append(
            (
                result.video.name,
                result.status.value,
                "-" if result.confidence is None else f"{result.confidence:.2f}",
                result.episode.code if result.episode else "-",
                result.episode.title if result.episode else "-",
                str(hit.distance) if hit else "-",
                str(result.supporting_stills) if result.supporting_stills else "-",
                format_timestamp(hit.timestamp if hit else None),
                result.runner_up.code if result.runner_up else "-",
                "; ".join(result.notes),
            )
        )
    return rows


#: Columns dropped entirely when no row has anything to say in them, which
#: keeps the table readable in a narrow terminal.
_OPTIONAL_COLUMNS = frozenset({"Title", "Conf", "Stills", "Runner-up", "Notes"})
_EMPTY_CELLS = frozenset({"", "-"})


def informative_columns(rows: Sequence[tuple[str, ...]]) -> list[int]:
    """Return the indices of the columns worth printing for ``rows``.

    Examples
    --------
    >>> len(informative_columns([]))
    5
    """
    keep = []
    for index, heading in enumerate(TABLE_COLUMNS):
        populated = any(row[index] not in _EMPTY_CELLS for row in rows)
        if heading not in _OPTIONAL_COLUMNS or populated:
            keep.append(index)
    return keep


def render_table(results: Sequence[MatchResult]) -> Table:
    """Build the console table summarising a run."""
    rows = table_rows(results)
    columns = informative_columns(rows)

    table = Table(title="MKV episode matches", show_lines=False, expand=False)
    for index in columns:
        heading = TABLE_COLUMNS[index]
        justify = "right" if heading in {"Conf", "Dist", "Stills", "Found at"} else "left"
        table.add_column(heading, overflow="fold", justify=justify)

    for result, row in zip(results, rows, strict=True):
        style = _STATUS_STYLE[result.status]
        styled = list(row)
        styled[1] = f"[{style}]{row[1]}[/{style}]"
        table.add_row(*(styled[index] for index in columns))
    return table


def export_previews(
    results: Sequence[MatchResult],
    *,
    still_paths: dict[str, Path],
    destination: Path,
    height: int = 360,
) -> list[Path]:
    """Write a side-by-side "still vs matched frame" image per result with a hit.

    Ambiguous results benefit most: seeing the two images next to each other
    settles in a glance what a Hamming distance only hints at.

    Parameters
    ----------
    results
        Results to illustrate. Those without a hit are skipped.
    still_paths
        Map of still URL to the downloaded image on disk.
    destination
        Directory to write previews into; created if missing.
    height
        Height of the composed preview in pixels.

    Returns
    -------
    list of pathlib.Path
        The previews that were written.
    """
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for result in results:
        hit = result.hit
        if hit is None:
            continue
        still_path = still_paths.get(hit.still.url)
        if still_path is None or not Path(still_path).exists():
            logger.debug("no local still for %s; skipping preview", hit.still.url)
            continue
        try:
            frame = grab_frame(result.video.path, hit.timestamp)
            with Image.open(still_path) as still:
                composed = _compose(still.convert("RGB"), frame.convert("RGB"), height)
        except (OSError, ValueError) as error:
            logger.warning("could not build preview for %s: %s", result.video.name, error)
            continue

        label = result.episode.code if result.episode else "candidate"
        target = destination / f"{result.video.path.stem}-{label}.jpg"
        composed.save(target, quality=88)
        written.append(target)
        logger.debug("wrote preview %s", target)

    return written


def _compose(left: Image.Image, right: Image.Image, height: int) -> Image.Image:
    """Place two images side by side at a common height, separated by a gutter."""
    gutter = 8
    scaled = [
        image.resize(
            (max(1, round(image.width * height / image.height)), height), Image.Resampling.LANCZOS
        )
        for image in (left, right)
    ]
    canvas = Image.new("RGB", (sum(image.width for image in scaled) + gutter, height), (16, 16, 16))
    canvas.paste(scaled[0], (0, 0))
    canvas.paste(scaled[1], (scaled[0].width + gutter, 0))
    return canvas
