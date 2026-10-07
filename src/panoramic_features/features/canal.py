from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from panoramic_features.geometry import mask_points, principal_axis

MIN_BINS = 30
TRIM = 0.10  # fraction of the centreline discarded at each end (polygon end caps)
ANGLE_STEP = 10  # centreline samples between the two points of a local slope
WIDTH_PERCENTILES = (5, 95)


def _centerline(canal: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Offset from the canal axis and canal width at each position along the axis."""
    skeleton = skeletonize(canal)
    points = mask_points(skeleton)
    if len(points) < MIN_BINS:
        return None

    center, axis = principal_axis(points)
    normal = np.array([-axis[1], axis[0]])
    t = (points - center) @ axis
    offset = (points - center) @ normal
    width = 2.0 * cv2.distanceTransform(canal.astype(np.uint8), cv2.DIST_L2, 5)[skeleton]

    bins = np.round(t).astype(int)
    bins -= bins.min()
    counts = np.bincount(bins)
    filled = counts > 0
    mean_offset = np.bincount(bins, weights=offset)[filled] / counts[filled]
    mean_width = np.bincount(bins, weights=width)[filled] / counts[filled]

    cut = int(TRIM * mean_offset.size)
    mean_offset = mean_offset[cut : mean_offset.size - cut]
    mean_width = mean_width[cut : mean_width.size - cut]
    return (mean_offset, mean_width) if mean_offset.size >= MIN_BINS * (1 - 2 * TRIM) else None


def canal_narrowing(canal: np.ndarray) -> float:
    """Relative loss of canal width along its centreline (0 = uniform width)."""
    line = _centerline(canal)
    if line is None:
        return float("nan")
    low, high = np.percentile(line[1], WIDTH_PERCENTILES)
    return float(1.0 - low / high) if high > 0 else float("nan")


def canal_deviation(canal: np.ndarray) -> float:
    """Standard deviation (degrees) of the local direction of the canal centreline."""
    line = _centerline(canal)
    if line is None or line[0].size <= 2 * ANGLE_STEP:
        return float("nan")
    offset = line[0]
    angles = np.degrees(np.arctan2(offset[ANGLE_STEP:] - offset[:-ANGLE_STEP], ANGLE_STEP))
    return float(angles.std())


def canal_discontinuity(tooth: np.ndarray, canal: np.ndarray) -> float:
    """Fraction of the canal outline that runs through the tooth region."""
    contours, _ = cv2.findContours(canal.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = np.zeros(canal.shape, np.uint8)
    cv2.drawContours(outline, contours, -1, 1, 1)
    total = int(outline.sum())
    return float(np.count_nonzero(outline.astype(bool) & tooth) / total) if total else float("nan")
