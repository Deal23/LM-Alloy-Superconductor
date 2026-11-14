"""Regression checks for package entry points, relocated paths and material scope."""
import subprocess
import sys

import pandas as pd
import pytest

from lmsc.dataset import build_frame
from lmsc.paths import DATASET, MODELS, ROOT


@pytest.mark.parametrize("formula, kept", [
    ("Nb3Sn", True),
    ("MgB2", True),
    ("FeSe", True),
    ("Nb", True),
    ("LaH10", False),
    ("SrTiO3", False),
    ("B", False),
])
def test_material_scope(formula, kept):
    frame, _ = build_frame(pd.DataFrame({"formula": [formula], "Tc": [1.0]}))
    assert bool(len(frame)) is kept


def test_relocated_assets_exist():
    assert DATASET.is_file()
    assert (MODELS / "feature_columns.json").is_file()


@pytest.mark.parametrize("args", [
    ["--help"], ["build", "--help"], ["predict", "--help"],
    ["scan", "--help"], ["anomaly", "--help"], ["elements", "--help"],
    ["space", "--help"], ["screening", "--help"], ["verify"],
])
def test_package_entry_point(args):
    result = subprocess.run([sys.executable, "-m", "lmsc", *args], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
