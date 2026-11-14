from lmsc.analysis import element_space


def test_classify_family_is_priority_and_mutually_exclusive():
    assert element_space.classify_family(frozenset({"Fe", "Ni"})) == "Fe-based"
    assert element_space.classify_family(frozenset({"Fe", "As"})) == "Fe-based"
    assert element_space.classify_family(frozenset({"Nb", "Sn"})) == "Alloy"
    assert element_space.classify_family(frozenset({"Mg", "B"})) == "Metal-nonmetal"


def test_parse_elements_drops_zero_coefficients():
    elems = element_space.parse_elements("Fe0.000Mo1.000")
    assert elems == frozenset({"Mo"})


def test_family_counts_are_disjoint_and_sum_to_rows():
    df = element_space.load_data(element_space.SOURCE)
    counts = df["family"].value_counts()
    assert len(df) == 12290
    assert counts.sum() == len(df)
    assert counts["Fe-based"] == 1636
    assert counts["Alloy"] == 4579
    assert counts["Metal-nonmetal"] == 6075
