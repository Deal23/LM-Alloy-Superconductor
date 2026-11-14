
import pandas as pd

from lmsc import dataset as build_dataset
from lmsc.paths import DATA, DATASET


def test_source_formula_is_not_converted_again():
    built, _ = build_dataset.build_frame(pd.DataFrame({"formula": ["Nb75Ga15Al10"], "Tc": [1]}))
    assert built.loc[0, "formula"] == "Nb75Ga15Al10"
    assert built.loc[0, "Nb"] == 0.75


def test_actual_build_matches_frozen_final_table():
    source = pd.read_csv(DATA / "source.csv")
    final = pd.read_csv(DATASET)
    built, dropped = build_dataset.build_frame(source)
    assert dropped["reason"].value_counts().to_dict() == {"contains_O_or_H": 8563, "no_metal": 122, "amorphous_gallium": 9, "fese_thin_film": 6, "high_pressure": 3, "disputed_superconductivity": 1, "uncertain_composition": 1}
    assert len(built) == len(final) == 12290
    assert built["formula"].nunique() == final["formula"].nunique() == 8076
    pd.testing.assert_frame_equal(built, final)


def test_audit_records_exclusions_without_changing_data():
    source = pd.DataFrame({"formula": ["Nb75Ga15Al10", "FeSe", "H2O", "Ga", "Xx"],
                           "Tc": [10, 8, 0, 8, 0]})
    audit = []
    actual, dropped = build_dataset.build_frame(source, audit=audit)
    expected, _ = build_dataset.build_frame(source)
    pd.testing.assert_frame_equal(actual, expected)
    assert len(audit) == len(source)
    assert [row["source_row"] for row in audit] == list(range(len(source)))
    assert audit[0]["status"] == "kept"
    assert audit[1]["status"] == "kept"
    assert audit[2]["reason"] == "contains_O_or_H"
    assert audit[3]["reason"] == "amorphous_gallium"
    assert audit[4]["reason"].startswith("invalid:")
    assert sum(row["status"] == "dropped" for row in audit) == len(dropped)
