
import pytest

from lmsc.composition import compositions, parse_formula


def test_formula_order_is_canonical_and_normalized():
    assert parse_formula("Sn0.5In0.5") == parse_formula("In0.5Sn0.5")
    assert sum(parse_formula("GaIn20.5Sn12.5").values()) == pytest.approx(1)


@pytest.mark.parametrize("formula", ["", "Fe0", "Fe-1", "FeNaN", "Xx1"])
def test_invalid_formula_is_rejected(formula):
    with pytest.raises(ValueError):
        parse_formula(formula, allowed={"Fe", "Na"})


def test_candidate_counts_and_uniqueness():
    candidates = list(compositions(["In", "Nb", "Sn"], 3, 4))
    assert len(candidates) == 3  # C(3, 3) * C(3, 2)
    assert len(set(candidates)) == len(candidates)
    assert all(sum(counts) == 4 for _, counts in candidates)


def test_candidate_count_changes_with_denominator():
    small = len(list(compositions(["In", "Nb", "Sn"], 2, 4)))
    large = len(list(compositions(["In", "Nb", "Sn"], 2, 8)))
    assert small == 3 * 3
    assert large == 3 * 7
    assert large > small
