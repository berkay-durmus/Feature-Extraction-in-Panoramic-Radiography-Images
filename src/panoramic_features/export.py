"""Writing the feature table to XLSX or CSV."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import xlsxwriter

from panoramic_features.pipeline import FEATURES, Row, is_missing

HEADERS = (
    "ImageNumber",
    "Tooth",
    "Obscuration",
    "Deflection",
    "NarrowingRoots",
    "NarrowingCanal",
    "DiscontinuityCanal",
    "CanalDeviation",
)


def _cells(row: Row, missing: float | None) -> list:
    values = [missing if is_missing(row.values[f]) else row.values[f] for f in FEATURES]
    return [row.image_id, row.tooth, *values]


def write_table(rows: Iterable[Row], path: str | Path, missing: float | None = None) -> int:
    """Write rows to ``path`` (format from the suffix). Missing values become ``missing``."""
    path = Path(path)
    table = [_cells(row, missing) for row in rows]

    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADERS)
            writer.writerows([["" if c is None else c for c in r] for r in table])
    elif path.suffix.lower() == ".xlsx":
        book = xlsxwriter.Workbook(str(path))
        sheet = book.add_worksheet("Features")
        sheet.write_row(0, 0, HEADERS)
        for r, cells in enumerate(table, 1):
            for c, cell in enumerate(cells):
                if cell is not None:
                    sheet.write(r, c, cell)
        book.close()
    else:
        raise ValueError(f"Unsupported output format: {path.suffix} (use .xlsx or .csv)")
    return len(table)
