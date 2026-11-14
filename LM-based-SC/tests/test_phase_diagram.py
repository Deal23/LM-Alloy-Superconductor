
import numpy as np
import pytest

from lmsc.discovery import predict


def test_binary_grid_covers_simplex():
    grid = list(predict.binary_compositions(("Nb", "Ga"), 4))
    assert len(grid) == 5
    assert grid[0][1] == (4, 0)
    assert grid[-1][1] == (0, 4)
    assert all(sum(counts) == 4 for _, counts in grid)


def test_ternary_grid_is_complete_and_unique():
    grid = list(predict.ternary_compositions(("Ga", "In", "Ag"), 6))
    assert len(grid) == 28
    assert len({counts for _, counts in grid}) == 28
    assert all(sum(counts) == 6 for _, counts in grid)


def test_barycentric_vertices():
    x, y = predict.barycentric_xy(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    assert x.tolist() == pytest.approx([0.0, 1.0, 0.5])
    assert y.tolist() == pytest.approx([0.0, 0.0, np.sqrt(3) / 2])


def test_formula_from_counts_skips_zero_and_normalises():
    assert predict.formula_from_counts(("Nb", "Ga"), (3, 2)) == "Nb0.6Ga0.4"
    assert predict.formula_from_counts(("Ga", "In", "Ag"), (6, 0, 0)) == "Ga"
