"""Composition utilities for alloys and metal–nonmetal compounds without H/O.

Single source of truth for element definitions, formula parsing/normalization,
and deterministic binary/ternary candidate generation.
"""
import itertools
import re

EXCLUDED = frozenset({"O", "H"})
ELEMENTS = sorted(
    "Ag Al Am As Au B Ba Be Bi Br C Ca Cd Ce Cl Co Cr Cs Cu Dy Er Eu F Fe Ga Gd Ge "
    "Hf Hg Ho I In Ir K La Li Lu Mg Mn Mo N Na Nb Nd Ni Np Os P Pa Pb Pd Pr Pt Pu "
    "Rb Re Rh Ru S Sb Sc Se Si Sm Sn Sr Ta Tb Tc Te Th Ti Tl Tm U V W Y Yb Zn Zr".split()
)
SOURCE_ELEMENTS = frozenset(ELEMENTS) | EXCLUDED
NONMETALS = frozenset({"B", "C", "N", "F", "Si", "P", "S", "Cl", "Ge", "As", "Se", "Br", "Sb", "Te", "I"})
METALS = frozenset(ELEMENTS) - NONMETALS
TARGET = "Tc_K"

TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?")


def parse_formula(formula, allowed=None, allow_zero=False):
    """Parse a formula into normalized atomic fractions.

    allowed: element set accepted by the parser (default: ELEMENTS, no O/H).
    allow_zero: tolerate zero coefficients (dropped after parsing).
    """
    text = str(formula).strip().rstrip(".")
    universe = frozenset(allowed) if allowed is not None else frozenset(ELEMENTS)
    counts, pos = {}, 0
    for match in TOKEN_RE.finditer(text):
        if match.start() != pos:
            raise ValueError(f"invalid formula: {formula!r}")
        element, number = match.groups()
        if element not in universe:
            raise ValueError(f"unknown element: {element}")
        value = float(number) if number else 1.0
        if value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"coefficient must be positive: {element}")
        counts[element] = counts.get(element, 0.0) + value
        pos = match.end()
    counts = {element: value for element, value in counts.items() if value > 0}
    if not counts or pos != len(text):
        raise ValueError(f"invalid formula: {formula!r}")
    total = sum(counts.values())
    return dict(sorted((element, value / total) for element, value in counts.items()))


def canonical_formula(elements, counts):
    return "".join(
        element + (str(int(count)) if int(count) != 1 else "")
        for element, count in zip(elements, counts)
    )


def compositions(elements, n_components, denominator):
    """Yield unique element/count candidates in deterministic order."""
    if n_components not in (2, 3) or denominator < n_components:
        raise ValueError("components must be 2 or 3 and denominator >= components")
    for selected in itertools.combinations(sorted(elements), n_components):
        for cuts in itertools.combinations(range(1, denominator), n_components - 1):
            points = (0,) + cuts + (denominator,)
            counts = tuple(points[i + 1] - points[i] for i in range(n_components))
            yield selected, counts


def feature_row(elements, counts, columns):
    total = float(sum(counts))
    fractions = dict(zip(elements, (count / total for count in counts)))
    return [fractions.get(column, 0.0) for column in columns]
