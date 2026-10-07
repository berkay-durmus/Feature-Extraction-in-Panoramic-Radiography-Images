"""Geometric helpers shared by the feature extractors."""

from __future__ import annotations

import cv2
import numpy as np

HORIZONTAL_THRESHOLD = 0.35  # |dy| below this means the tooth axis is close to horizontal


def principal_axis(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unit direction of the dominant axis of ``(N, 2)`` points."""
    center = points_xy.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points_xy - center).T))
    return center, vectors[:, -1]


def mask_points(mask: np.ndarray) -> np.ndarray:
    """``(N, 2)`` array of ``(x, y)`` coordinates of the true pixels."""
    ys, xs = np.nonzero(mask)
    return np.column_stack([xs, ys]).astype(float)


def rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a boolean mask by ``angle`` radians (image coordinates), enlarging the canvas."""
    h, w = mask.shape
    cos, sin = np.cos(angle), np.sin(angle)
    matrix = np.array([[cos, -sin, 0.0], [sin, cos, 0.0]])
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)
    rotated = corners @ matrix[:, :2].T
    low, high = rotated.min(axis=0), rotated.max(axis=0)
    matrix[:, 2] = -low
    size = tuple(int(np.ceil(v)) for v in high - low)
    out = cv2.warpAffine(mask.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
    return out.astype(bool)


def crop_to_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def orient_tooth(tooth: np.ndarray, side: str) -> np.ndarray:
    """Rotate a tooth mask so its long axis is vertical with the roots pointing down.

    The root end is the lower one; for near-horizontal (impacted) teeth it is the end
    pointing away from the midline, i.e. image-left for the right tooth and image-right
    for the left tooth.
    """
    points = mask_points(tooth)
    _, direction = principal_axis(points)
    dx, dy = direction
    if abs(dy) < HORIZONTAL_THRESHOLD:
        outward = -1.0 if side == "right" else 1.0
        if dx * outward < 0:
            dx, dy = -dx, -dy
    elif dy < 0:
        dx, dy = -dx, -dy
    angle = np.pi / 2 - np.arctan2(dy, dx)
    return crop_to_mask(rotate_mask(tooth, angle))
