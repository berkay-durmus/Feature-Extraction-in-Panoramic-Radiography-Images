import csv

import cv2
import numpy as np
import pytest

from conftest import SHAPE, write_xml
from panoramic_features.cli import main
from panoramic_features.pipeline import FEATURES, extract_features, find_pairs

TOOTH = [(130, 120), (230, 120), (230, 330), (130, 330)]
CANAL = [(60, 250), (450, 250), (450, 290), (60, 290)]


@pytest.fixture
def dataset(tmp_path, rng):
    for name in ("2", "10"):
        image = rng.integers(60, 250, SHAPE).astype(np.uint8)
        cv2.imwrite(str(tmp_path / f"{name}.jpg"), image)
        write_xml(tmp_path / f"{name}.xml", {"48": TOOTH, "Sağ M3": CANAL})
    cv2.imwrite(str(tmp_path / "orphan.jpg"), np.zeros(SHAPE, np.uint8))
    return tmp_path


def test_find_pairs_matches_stems_in_natural_order(dataset):
    assert [i.stem for i, _ in find_pairs(dataset)] == ["2", "10"]


def test_missing_left_tooth_gives_nan_not_error(dataset):
    rows = extract_features(dataset / "2.jpg", dataset / "2.xml")
    right, left = rows
    assert (right.tooth, left.tooth) == ("48", "38")
    assert set(right.values) == set(FEATURES)
    assert not np.isnan(right.values["canal_discontinuity"])
    assert all(np.isnan(v) for v in left.values.values())


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_cli_writes_table(dataset, tmp_path, suffix):
    out = tmp_path / f"out{suffix}"
    assert main([str(dataset), "-o", str(out), "--missing", "-1"]) == 0
    if suffix == ".csv":
        rows = list(csv.reader(out.open(encoding="utf-8")))
    else:
        from openpyxl import load_workbook

        rows = [[c for c in r] for r in load_workbook(out).active.values]
    assert rows[0][:2] == ["ImageNumber", "Tooth"] or list(rows[0][:2]) == ["ImageNumber", "Tooth"]
    assert len(rows) == 5  # header + 2 images x 2 teeth
    assert str(rows[2][2]) in ("-1", "-1.0")


def test_cli_fails_on_empty_folder(tmp_path):
    assert main([str(tmp_path)]) == 1
    assert main([str(tmp_path / "nope")]) == 2
