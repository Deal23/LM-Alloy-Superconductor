
import pandas as pd
import pytest

from lmsc.composition import ELEMENTS, METALS, TARGET, parse_formula
from lmsc.paths import DATASET as DATA


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


def test_columns_and_normalization(df):
    assert list(df.columns) == ["formula", *ELEMENTS, TARGET]
    assert df[ELEMENTS].sum(axis=1).sub(1.0).abs().max() < 1e-9


def test_no_oxygen_or_hydrogen(df):
    for formula in df["formula"]:
        assert not ({"O", "H"} & parse_formula(formula, allow_zero=True).keys())


def test_every_row_contains_a_metal(df):
    for formula in df["formula"]:
        assert METALS & parse_formula(formula, allow_zero=True).keys()


def test_target_is_valid(df):
    assert df[TARGET].notna().all()
    assert (df[TARGET] >= 0).all()


def test_elemental_ga_is_crystalline(df):
    ga = df[df["Ga"].sub(1.0).abs() < 1e-9]
    assert len(ga) == 4
    assert ga[TARGET].between(0.5, 2.0).all()
