"""Reading polygon annotations from the XML files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

Polygon = list[tuple[int, int]]

RIGHT_TOOTH, RIGHT_CANAL = "48", "right canal"
LEFT_TOOTH, LEFT_CANAL = "38", "left canal"

# Accepted spellings (case-insensitive) mapped to the canonical labels above.
ALIASES = {
    "right canal": RIGHT_CANAL,
    "right_canal": RIGHT_CANAL,
    "right m3": RIGHT_CANAL,
    "sağ m3": RIGHT_CANAL,
    "left canal": LEFT_CANAL,
    "left_canal": LEFT_CANAL,
    "left m3": LEFT_CANAL,
    "sol m3": LEFT_CANAL,
}


def canonical_label(name: str) -> str:
    name = name.strip()
    return ALIASES.get(name.casefold(), name)


def parse_annotation(path: str | Path) -> dict[str, list[Polygon]]:
    """Return ``{label: [polygon, ...]}`` with canonical labels; a polygon is ``[(x, y), ...]``."""
    root = ET.parse(path).getroot()
    polygons: dict[str, list[Polygon]] = defaultdict(list)

    for item in root.iterfind("./outputs/object/item"):
        label = item.findtext("name")
        node = item.find("polygon")
        if label is None or node is None:
            continue

        points: Polygon = []
        index = 1
        while True:
            x, y = node.findtext(f"x{index}"), node.findtext(f"y{index}")
            if x is None or y is None:
                break
            points.append((round(float(x)), round(float(y))))
            index += 1

        if len(points) >= 3:
            polygons[canonical_label(label)].append(points)

    return dict(polygons)
