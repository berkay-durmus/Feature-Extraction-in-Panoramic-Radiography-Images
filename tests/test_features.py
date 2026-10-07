import numpy as np
import pytest

from conftest import SHAPE, fill, tooth_mask
from panoramic_features.features import (
    canal_deviation,
    canal_discontinuity,
    canal_narrowing,
    obscuration,
    root_deflection,
    root_narrowing,
)


def band(top_edge, bottom_edge, x0=50, x1=450):
    xs = np.arange(x0, x1 + 1, 5)
    upper = [(int(x), int(top_edge(x))) for x in xs]
    lower = [(int(x), int(bottom_edge(x))) for x in xs[::-1]]
    return fill(upper + lower)


def test_obscuration_positive_when_overlap_is_dark(rng):
    tooth = tooth_mask()
    canal = fill([(120, 250), (240, 250), (240, 340), (120, 340)])
    image = rng.integers(150, 250, SHAPE).astype(np.uint8)
    image[tooth & canal] = 60
    assert obscuration(image, tooth, canal) > 0.3


def test_obscuration_nan_without_overlap(rng):
    tooth = tooth_mask()
    canal = fill([(300, 300), (400, 300), (400, 350), (300, 350)])
    image = rng.integers(100, 250, SHAPE).astype(np.uint8)
    assert np.isnan(obscuration(image, tooth, canal))


@pytest.mark.parametrize("side", ["right", "left"])
def test_straight_roots_have_no_deflection(side):
    assert root_deflection(tooth_mask(), side) < 5


def test_bent_root_is_detected():
    bent = ((140, 200), (170, 200), (170, 270), (110, 330), (80, 310), (140, 260))
    straight = root_deflection(tooth_mask(), "right")
    assert root_deflection(tooth_mask(left_root=bent), "right") > straight + 15


def test_deflection_is_rotation_invariant():
    import cv2

    mask = tooth_mask().astype(np.uint8)
    matrix = cv2.getRotationMatrix2D((250, 200), 25, 1.0)
    rotated = cv2.warpAffine(mask, matrix, SHAPE[::-1], flags=cv2.INTER_NEAREST).astype(bool)
    assert abs(root_deflection(rotated, "right") - root_deflection(tooth_mask(), "right")) < 8


def test_pinched_root_scores_higher_than_straight():
    pinched = (
        (140, 200),
        (175, 200),
        (175, 250),
        (160, 270),
        (175, 290),
        (175, 330),
        (140, 330),
        (140, 290),
        (155, 270),
        (140, 250),
    )
    assert (
        root_narrowing(tooth_mask(left_root=pinched), "right")
        > root_narrowing(tooth_mask(), "right") + 0.2
    )


def test_canal_narrowing_uniform_vs_tapered():
    uniform = band(lambda x: 200, lambda x: 230)
    tapered = band(lambda x: 200, lambda x: 200 + 4 + 30 * (450 - x) / 400)
    assert canal_narrowing(uniform) < 0.15
    assert canal_narrowing(tapered) > canal_narrowing(uniform) + 0.3


def test_canal_deviation_straight_vs_wavy():
    straight = band(lambda x: 200, lambda x: 230)
    wavy = band(lambda x: 200 + 25 * np.sin(x / 25), lambda x: 230 + 25 * np.sin(x / 25))
    assert canal_deviation(straight) < 2
    assert canal_deviation(wavy) > 10


def test_canal_features_nan_for_tiny_canal():
    tiny = fill([(10, 10), (14, 10), (14, 14), (10, 14)])
    assert np.isnan(canal_narrowing(tiny))
    assert np.isnan(canal_deviation(tiny))


def test_discontinuity_fraction():
    canal = fill([(100, 250), (400, 250), (400, 290), (100, 290)])
    assert canal_discontinuity(fill([(0, 0), (5, 0), (5, 5)]), canal) == 0
    everything = np.ones(SHAPE, bool)
    assert canal_discontinuity(everything, canal) == 1
    half = fill([(0, 200), (250, 200), (250, 400), (0, 400)])
    assert 0.3 < canal_discontinuity(half, canal) < 0.7
