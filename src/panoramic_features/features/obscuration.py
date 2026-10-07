from __future__ import annotations

import numpy as np
from skimage.filters import threshold_multiotsu

MIN_PIXELS = 20
CLASSES = 5
REFERENCE_CLASS = 2  # middle Otsu threshold is used as the reference tooth density


def obscuration(image: np.ndarray, tooth: np.ndarray, canal: np.ndarray) -> float:
    """Contrast between the root/canal overlap and the rest of the tooth.

    Positive when the overlap is darker than the reference tooth density.
    """
    overlap = image[tooth & canal]
    rest = image[tooth & ~canal]
    if overlap.size < MIN_PIXELS or rest.size < MIN_PIXELS or np.unique(rest).size < CLASSES:
        return float("nan")

    reference = float(threshold_multiotsu(rest, classes=CLASSES)[REFERENCE_CLASS])
    mean = float(overlap.mean())
    total = reference + mean
    return (reference - mean) / total if total > 0 else float("nan")
