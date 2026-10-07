from __future__ import annotations

import cv2
import numpy as np

from panoramic_features.geometry import crop_to_mask, mask_points, orient_tooth

MIN_ROOT_PIXELS = 40
MIN_ROOT_AREA_RATIO = 0.15  # a component smaller than this fraction of the largest is noise
APICAL_START = 0.5  # root separation is judged on this lower fraction of the root half
SMOOTH_WINDOW = 5
DROP_SPAN = 0.10  # width drop is measured over this fraction of the root length
DROP_SEARCH = 0.70  # ... starting within this fraction from the crown


def _split_roots(tooth: np.ndarray, side: str) -> list[np.ndarray]:
    """Orient the tooth, keep its lower half and separate it into at most two roots.

    The roots are told apart by the connected components of the apical part, where they
    are not yet joined; the whole lower half is then cut midway between them.
    """
    oriented = orient_tooth(tooth, side)
    half = oriented.shape[0] // 2
    lower = oriented.copy()
    lower[:half] = False
    if not lower.any():
        return []

    apical = lower.copy()
    apical[: half + int(APICAL_START * (oriented.shape[0] - half))] = False
    count, _, stats, centroids = cv2.connectedComponentsWithStats(apical.astype(np.uint8))
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i for i in np.argsort(-areas)[:2] if areas[i] >= MIN_ROOT_AREA_RATIO * areas.max()]

    if len(keep) < 2:
        return [crop_to_mask(lower)]

    cut = int(centroids[[i + 1 for i in keep], 0].mean())
    parts = [lower.copy(), lower.copy()]
    parts[0][:, cut:] = False
    parts[1][:, :cut] = False
    return [crop_to_mask(p) for p in parts if p.sum() >= MIN_ROOT_PIXELS]


def _bend(root: np.ndarray) -> float:
    """Angle (degrees) between the coronal-to-middle and middle-to-apical direction of a root."""
    points = mask_points(root)
    rows = np.quantile(points[:, 1], [1 / 3, 2 / 3])
    thirds = [
        points[points[:, 1] <= rows[0]],
        points[(points[:, 1] > rows[0]) & (points[:, 1] <= rows[1])],
        points[points[:, 1] > rows[1]],
    ]
    if min(len(t) for t in thirds) < MIN_ROOT_PIXELS // 3:
        return float("nan")

    first, middle, last = (t.mean(axis=0) for t in thirds)
    a, b = middle - first, last - middle
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b / norm, -1, 1)))) if norm else float("nan")


def _max_width_drop(root: np.ndarray) -> float:
    widths = root.sum(axis=1).astype(float)
    widths = widths[widths > 0]
    if widths.size < 3 * SMOOTH_WINDOW:
        return float("nan")
    widths = np.convolve(widths, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode="valid")

    span = max(2, int(DROP_SPAN * widths.size))
    starts = widths[: max(1, int(DROP_SEARCH * widths.size))]
    ends = widths[span : span + starts.size]
    n = min(starts.size, ends.size)
    return float(np.max((starts[:n] - ends[:n]) / starts[:n])) if n else float("nan")


def _aggregate(values: list[float]) -> float:
    valid = [v for v in values if not np.isnan(v)]
    return max(valid) if valid else float("nan")


def root_deflection(tooth: np.ndarray, side: str) -> float:
    """Largest bend angle (degrees) between the coronal and apical half of a root."""
    return _aggregate([_bend(r) for r in _split_roots(tooth, side) if r.sum() >= MIN_ROOT_PIXELS])


def root_narrowing(tooth: np.ndarray, side: str) -> float:
    """Largest relative width loss over a short stretch of a root (0 = none, 1 = pinched off)."""
    return _aggregate([_max_width_drop(r) for r in _split_roots(tooth, side)])
