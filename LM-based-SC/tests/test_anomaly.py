
import numpy as np
import pandas as pd

from lmsc.analysis import anomaly


def test_robust_residual_flags_large_tail():
    residual = np.array([0.1, -0.2, 0.0, 0.3, 12.0])
    abs_error = np.abs(residual)
    z, flag, center, mad = anomaly.robust_residual(
        residual, abs_error, z_threshold=3.5, min_abs_error=5.0
    )
    assert center == 0.1
    assert mad > 0
    assert flag.sum() == 1
    assert flag[-1]
    assert z[-1] > 3.5


def test_duplicate_group_stats_and_flag():
    groups = pd.Series(["A", "A", "B", "B", "B"])
    y = pd.Series([10.0, 30.0, 5.0, 5.5, 6.0])
    stats = anomaly.duplicate_stats(groups, y)
    a = stats[stats["group"] == "A"].iloc[0]
    assert a["group_count"] == 2
    assert a["group_range"] == 20.0

    table = pd.DataFrame({
        "Tc_K": [10.0, 30.0],
        "group_count": [2, 2],
        "group_median": [20.0, 20.0],
        "group_mad": [10.0, 10.0],
        "group_range": [20.0, 20.0],
    })
    table = anomaly.add_duplicate_flags(table, duplicate_range=10.0, min_abs_error=5.0)
    assert table["duplicate_flag"].all()
    assert table["group_deviation"].tolist() == [10.0, 10.0]
