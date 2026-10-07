"""Reading polygon annotations from the XML files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

Polygon = list[tuple[int, int]]

RIGHT_TOOTH, RIGHT_CANAL = "48", "Sağ M3"
LEFT_TOOTH, LEFT_CANAL = "38", "Sol M3"


def parse_annotation(path: str | Path) -> dict[str, list[Polygon]]:
    """Return ``{label: [polygon, ...]}``; a polygon is a list of ``(x, y)`` pixel points."""
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
            polygons[label.strip()].append(points)

    return dict(polygons)
