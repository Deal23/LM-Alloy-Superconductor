"""Build the alloy and metal–nonmetal dataset, excluding H/O-containing systems.

The source is already cleaned; retain its formulas and normalize atomic fractions.

Only the source table and the final table are kept in the repository:
    data/source.csv  ->  data/superconductors.csv

Usage:
    python -m lmsc build
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .composition import ELEMENTS, EXCLUDED, METALS, SOURCE_ELEMENTS, TARGET, parse_formula
from .paths import DATA, DATASET

SOURCE = DATA / "source.csv"
OUTPUT = DATASET

# Individually reviewed records; do not infer experimental conditions from Tc.
REVIEWED_EXCLUSIONS = {
    ("Fe1Se1", 40.0): "fese_thin_film",
    ("Yb1Fe4Al8", 50.0): "disputed_superconductivity",
    ("S2Ti1", 45.0): "uncertain_composition",
    ("Li1", 20.0): "high_pressure",
    ("Sc1", 19.6): "high_pressure",
    ("Ba1Fe2As2", 35.0): "high_pressure",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_frame(source, audit=None):
    records, dropped = [], []
    for row, (formula, tc) in enumerate(zip(source["formula"], source["Tc"])):
        entry = {"source_row": row, "source_formula": formula, TARGET: float(tc)}
        if audit is not None:
            audit.append(entry)
        try:
            fractions = parse_formula(formula, allowed=SOURCE_ELEMENTS, allow_zero=True)
        except ValueError as exc:
            entry.update(status="dropped", reason=f"invalid: {exc}")
            dropped.append({"formula": formula, TARGET: float(tc), "reason": f"invalid: {exc}"})
            continue
        entry.update(status="kept", reason="")
        if EXCLUDED & fractions.keys():
            entry.update(status="dropped", reason="contains_O_or_H")
            dropped.append({"formula": formula, TARGET: float(tc), "reason": "contains_O_or_H"})
            continue
        if not METALS & fractions.keys():
            entry.update(status="dropped", reason="no_metal")
            dropped.append({"formula": formula, TARGET: float(tc), "reason": "no_metal"})
            continue
        if set(fractions) == {"Ga"} and float(tc) > 2.0:
            entry.update(status="dropped", reason="amorphous_gallium")
            dropped.append({"formula": formula, TARGET: float(tc), "reason": "amorphous_gallium"})
            continue
        # MDR identifies these FeSe measurements as single-layer film results.
        if formula == "Fe1Se1" and float(tc) in {55.0, 60.0, 65.0, 80.0}:
            entry.update(status="dropped", reason="fese_thin_film")
            dropped.append({"formula": formula, TARGET: float(tc), "reason": "fese_thin_film"})
            continue
        reason = REVIEWED_EXCLUSIONS.get((formula, float(tc)))
        if reason:
            entry.update(status="dropped", reason=reason)
            dropped.append({"formula": formula, TARGET: float(tc), "reason": reason})
            continue
        records.append(
            {
                "formula": formula,
                **{element: fractions.get(element, 0.0) for element in ELEMENTS},
                TARGET: float(tc),
            }
        )
    out = pd.DataFrame(records, columns=["formula", *ELEMENTS, TARGET])
    if not (abs(out[ELEMENTS].sum(axis=1) - 1.0) < 1e-9).all():
        raise SystemExit("normalized fractions do not sum to 1")
    dropped_frame = pd.DataFrame(dropped, columns=["formula", TARGET, "reason"])
    return out, dropped_frame


def main():
    parser = argparse.ArgumentParser(description="Build data/superconductors.csv from the frozen source.")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit", type=Path, help="Optional row-level cleaning audit CSV (zero-based source rows)")
    args = parser.parse_args()

    source = pd.read_csv(args.source)
    missing = {"formula", "Tc"} - set(source.columns)
    if missing:
        raise SystemExit(f"source is missing columns: {sorted(missing)}")
    if source["formula"].isna().any() or source["Tc"].isna().any():
        raise SystemExit("source contains missing formula or Tc")

    audit = [] if args.audit else None
    out, dropped = build_frame(source, audit=audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit).to_csv(args.audit, index=False)

    reasons = dropped["reason"].value_counts().to_dict() if len(dropped) else {}
    print(f"source rows        : {len(source):,}")
    print(f"dropped O/H        : {reasons.get('contains_O_or_H', 0):,}")
    print(f"dropped non-metal  : {reasons.get('no_metal', 0):,}")
    print(f"dropped amorphous Ga: {reasons.get('amorphous_gallium', 0):,}")
    print(f"dropped FeSe films : {reasons.get('fese_thin_film', 0):,}")
    print(f"dropped reviewed   : {sum(reasons.get(key, 0) for key in ('disputed_superconductivity', 'uncertain_composition', 'high_pressure')):,}")
    print(f"dropped invalid    : {sum(count for reason, count in reasons.items() if reason.startswith('invalid:')):,}")
    print(f"final rows         : {len(out):,}")
    print(f"unique formulas    : {out['formula'].nunique():,}")
    print(f"sha256             : {sha256(args.output)}")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
