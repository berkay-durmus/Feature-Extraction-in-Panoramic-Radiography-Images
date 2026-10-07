"""Image loading and mask helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from panoramic_features.annotations import Polygon


def load_gray(path: str | Path) -> np.ndarray:
    """Load an image as 8-bit grayscale (unicode-safe paths)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def polygons_to_mask(shape: tuple[int, int], polygons: list[Polygon] | None) -> np.ndarray | None:
    """Rasterise polygons into a boolean mask; ``None`` if there is nothing inside the image."""
    if not polygons:
        return None
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(p, dtype=np.int32) for p in polygons], 1)
    return mask.astype(bool) if mask.any() else None
