import importlib.util
import sys
from pathlib import Path

import numpy as np

from panoramic_features.cli import main

SCRIPT = Path(__file__).parent.parent / "scripts" / "make_dummy_dataset.py"


def test_dummy_dataset_runs_through_the_pipeline(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("make_dummy_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path / "d"), "--count", "2"])
    module.main()

    out = tmp_path / "out.csv"
    assert main([str(tmp_path / "d"), "-o", str(out), "--missing", "nan"]) == 0
    lines = out.read_text().splitlines()[1:]
    assert len(lines) == 4
    assert not any(np.isnan(float(v)) for line in lines for v in line.split(",")[2:])
