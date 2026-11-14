
import pandas as pd

from lmsc.analysis import element_frequency
from lmsc.composition import ELEMENTS


def test_periodic_layout_covers_all_elements_once():
    assert len(element_frequency.POSITIONS) == 118
    assert set(element_frequency.POSITIONS) == set(element_frequency.SYMBOLS)
    assert len(set(element_frequency.POSITIONS.values())) == 118
    assert element_frequency.POSITIONS["H"] == (1, 1)
    assert element_frequency.POSITIONS["Lu"] == (17, 8.8)
    assert element_frequency.POSITIONS["Lr"] == (17, 9.8)


def test_count_present_counts_each_formula_once():
    frame = pd.DataFrame(0.0, index=range(3), columns=ELEMENTS)
    frame.loc[0, ["Nb", "Sn"]] = [0.75, 0.25]
    frame.loc[1, ["Nb"]] = [1.0]
    frame.loc[2, ["Ga", "In", "Sn"]] = [0.4, 0.4, 0.2]

    counts = element_frequency.count_present(frame)
    assert counts.sum() == 6
    assert counts["Nb"] == 2
    assert counts["Sn"] == 2
    assert counts["Ga"] == 1
    assert counts["Fe"] == 0


def test_frequency_table_has_counts_fractions_and_atomic_number():
    frame = pd.DataFrame(0.0, index=range(2), columns=ELEMENTS)
    frame.loc[0, ["Nb", "Sn"]] = [0.75, 0.25]
    frame.loc[1, ["Nb"]] = [1.0]

    table = element_frequency.build_frequency_table(frame).set_index("symbol")
    assert table.loc["Nb", "count"] == 2
    assert table.loc["Sn", "count"] == 1
    assert table.loc["Ga", "count"] == 0
    assert table.loc["Nb", "fraction"] == 1.0
    assert table.loc["Sn", "fraction"] == 0.5
    assert table.loc["Nb", "atomic_number"] == 41
    assert len(table) == 118
    assert table.loc["H", "count"] == 0
