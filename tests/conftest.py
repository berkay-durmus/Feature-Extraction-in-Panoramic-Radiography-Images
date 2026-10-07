import cv2
import numpy as np
import pytest

SHAPE = (400, 500)


def fill(*polygons, shape=SHAPE):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.array(p, np.int32) for p in polygons], 1)
    return mask.astype(bool)


def tooth_mask(
    left_root=((140, 200), (170, 200), (170, 330), (140, 330)),
    right_root=((190, 200), (220, 200), (220, 330), (190, 330)),
):
    crown = ((130, 120), (230, 120), (230, 200), (130, 200))
    return fill(crown, left_root, right_root)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def write_xml(path, items):
    """items: {label: [(x, y), ...]}"""
    parts = ["<annotation><outputs><object>"]
    for label, points in items.items():
        coords = "".join(f"<x{i}>{x}</x{i}><y{i}>{y}</y{i}>" for i, (x, y) in enumerate(points, 1))
        parts.append(f"<item><name>{label}</name><polygon>{coords}</polygon></item>")
    parts.append("</object></outputs></annotation>")
    path.write_text("".join(parts), encoding="utf-8")
