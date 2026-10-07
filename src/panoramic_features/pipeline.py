"""Batch feature extraction over a folder of images and annotations."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from panoramic_features import annotations as ann
from panoramic_features.features import (
    canal_deviation,
    canal_discontinuity,
    canal_narrowing,
    obscuration,
    root_deflection,
    root_narrowing,
)
from panoramic_features.imaging import load_gray, polygons_to_mask

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FEATURES = (
    "obscuration",
    "root_deflection",
    "root_narrowing",
    "canal_narrowing",
    "canal_discontinuity",
    "canal_deviation",
)

SIDES = (
    ("48", "right", ann.RIGHT_TOOTH, ann.RIGHT_CANAL),
    ("38", "left", ann.LEFT_TOOTH, ann.LEFT_CANAL),
)


@dataclass(frozen=True)
class Row:
    image_id: str
    tooth: str
    values: dict[str, float]


def _natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.stem)]


def find_pairs(folder: str | Path) -> list[tuple[Path, Path]]:
    """Match images with the XML file sharing their file name stem."""
    folder = Path(folder)
    images = sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key
    )
    xmls = {p.stem: p for p in folder.glob("*.xml")}

    pairs = []
    for image in images:
        if image.stem in xmls:
            pairs.append((image, xmls.pop(image.stem)))
        else:
            log.warning("No annotation for %s, skipped", image.name)
    for stem in xmls:
        log.warning("No image for annotation %s.xml, skipped", stem)
    return pairs


def _safe(name: str, label: str, func: Callable[..., float], *args) -> float:
    try:
        return float(func(*args))
    except Exception:
        log.warning("%s failed for %s", name, label, exc_info=True)
        return float("nan")


def extract_features(image_path: str | Path, xml_path: str | Path) -> list[Row]:
    """Compute all features for teeth 48 and 38 of one radiograph."""
    image = load_gray(image_path)
    polygons = ann.parse_annotation(xml_path)
    image_id = Path(image_path).stem

    rows = []
    for tooth_name, side, tooth_label, canal_label in SIDES:
        tooth = polygons_to_mask(image.shape, polygons.get(tooth_label))
        canal = polygons_to_mask(image.shape, polygons.get(canal_label))
        label = f"{image_id}/{tooth_name}"
        nan = float("nan")
        both = tooth is not None and canal is not None

        values = {
            "obscuration": _safe("obscuration", label, obscuration, image, tooth, canal)
            if both
            else nan,
            "root_deflection": _safe("root_deflection", label, root_deflection, tooth, side)
            if tooth is not None
            else nan,
            "root_narrowing": _safe("root_narrowing", label, root_narrowing, tooth, side)
            if tooth is not None
            else nan,
            "canal_narrowing": _safe("canal_narrowing", label, canal_narrowing, canal)
            if canal is not None
            else nan,
            "canal_discontinuity": _safe(
                "canal_discontinuity", label, canal_discontinuity, tooth, canal
            )
            if both
            else nan,
            "canal_deviation": _safe("canal_deviation", label, canal_deviation, canal)
            if canal is not None
            else nan,
        }
        rows.append(Row(image_id, tooth_name, values))
    return rows


def process_folder(folder: str | Path) -> Iterator[Row]:
    pairs = find_pairs(folder)
    for index, (image_path, xml_path) in enumerate(pairs, 1):
        log.info("[%d/%d] %s", index, len(pairs), image_path.name)
        try:
            yield from extract_features(image_path, xml_path)
        except Exception:
            log.error("Skipping %s", image_path.name, exc_info=True)


def is_missing(value: float) -> bool:
    return bool(np.isnan(value))
