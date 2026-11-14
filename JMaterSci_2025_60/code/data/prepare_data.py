"""Notebook cleaning, duplicate inspection and explicit mass-to-atom conversion."""
import argparse
import csv
from functools import lru_cache

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, TABLES, parse_formula, write_csv, write_json


@lru_cache(maxsize=1)
def atomic_masses():
    with Path(__file__).with_name("atomic_masses.csv").open(newline="") as handle:
        return {row["symbol"]: float(row["atomic_mass"]) for row in csv.DictReader(handle)}


def clean_formula(formula, convert=True):
    counts = parse_formula(formula)
    masses = atomic_masses()
    unknown = set(counts) - set(masses)
    if unknown:
        raise ValueError(f"Unknown elements: {sorted(unknown)}")
    counts = {key: float(f"{value:.3f}") for key, value in counts.items()}
    converted = convert and round(sum(counts.values()), 0) == 100
    if converted:
        counts = {key: value / masses[key] for key, value in counts.items()}
        total = sum(counts.values())
        return "".join(f"{key}{value / total:.3f}" for key, value in counts.items()), True
    return "".join(f"{key}{value:.3f}".rstrip("0").rstrip(".")
                   for key, value in counts.items()), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=str(DATA / "source/mdr.csv"))
    parser.add_argument("--output", type=str, default=str(TABLES / "generated/mdr_clean.csv"))
    parser.add_argument("--no-mass-conversion", action="store_true")
    parser.add_argument("--average-duplicates", action="store_true")
    args = parser.parse_args()
    source = pd.read_csv(args.input)
    rows, rejected, n_converted = [], [], 0
    for index, row in source.iterrows():
        try:
            tc = float(row["Tc"])
            if not pd.notna(tc) or tc in (float("inf"), -float("inf")):
                raise ValueError("Non-finite Tc")
            formula, converted = clean_formula(row["formula"], not args.no_mass_conversion)
            rows.append({"formula": formula, "Tc": tc})
            n_converted += int(converted)
        except (ValueError, ZeroDivisionError) as exc:
            rejected.append({"source_row": index + 2, "formula": row["formula"], "reason": str(exc)})
    result = pd.DataFrame(rows)
    duplicates = result[result.duplicated("formula", keep=False)]
    if args.average_duplicates:
        result = result.groupby("formula", sort=False).agg(Tc=("Tc", "mean"), count=("Tc", "size")).reset_index()
    write_csv(result, args.output)
    from pathlib import Path
    output = Path(args.output)
    write_csv(duplicates, output.with_name(output.stem + "_duplicates.csv"))
    write_csv(pd.DataFrame(rejected, columns=["source_row", "formula", "reason"]),
              output.with_name(output.stem + "_rejected.csv"))
    write_json({"source_rows": len(source), "output_rows": len(result),
                "mass_conversion_heuristic_rows": n_converted, "rejected_rows": len(rejected),
                "note": "Rounded subscript sum 100 is a historical heuristic, not proof of weight units."},
               output.with_suffix(".json"))
    print(f"Saved {len(result)} rows to {output}")


if __name__ == "__main__":
    main()
