"""Periodic-table frequency of elements in the superconductors dataset.

Each formula contributes one count per element, regardless of stoichiometry.
Outputs a single minimal periodic-table heatmap and a count table.

Usage:
    python -m lmsc elements
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Rectangle

from ..composition import ELEMENTS
from ..paths import ROOT, DATASET, DATA_RESULTS

DATA_PATH = DATASET
TABLE_PATH = DATA_RESULTS / "element_frequency.csv"
FIGURE_PATH = DATA_RESULTS / "data_element_frequency.png"

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb "
    "Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
    "Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es "
    "Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ATOMIC_NUMBER = {symbol: number for number, symbol in enumerate(SYMBOLS, start=1)}

PERIOD_BLOCKS = {
    1: {1: "H", 18: "He"},
    2: {1: "Li", 2: "Be", 13: "B", 14: "C", 15: "N", 16: "O", 17: "F", 18: "Ne"},
    3: {1: "Na", 2: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar"},
    4: {
        1: "K", 2: "Ca", 3: "Sc", 4: "Ti", 5: "V", 6: "Cr", 7: "Mn", 8: "Fe",
        9: "Co", 10: "Ni", 11: "Cu", 12: "Zn", 13: "Ga", 14: "Ge", 15: "As",
        16: "Se", 17: "Br", 18: "Kr",
    },
    5: {
        1: "Rb", 2: "Sr", 3: "Y", 4: "Zr", 5: "Nb", 6: "Mo", 7: "Tc", 8: "Ru",
        9: "Rh", 10: "Pd", 11: "Ag", 12: "Cd", 13: "In", 14: "Sn", 15: "Sb",
        16: "Te", 17: "I", 18: "Xe",
    },
    6: {
        1: "Cs", 2: "Ba", 4: "Hf", 5: "Ta", 6: "W", 7: "Re", 8: "Os", 9: "Ir",
        10: "Pt", 11: "Au", 12: "Hg", 13: "Tl", 14: "Pb", 15: "Bi", 16: "Po",
        17: "At", 18: "Rn",
    },
    7: {
        1: "Fr", 2: "Ra", 4: "Rf", 5: "Db", 6: "Sg", 7: "Bh", 8: "Hs", 9: "Mt",
        10: "Ds", 11: "Rg", 12: "Cn", 13: "Nh", 14: "Fl", 15: "Mc", 16: "Lv",
        17: "Ts", 18: "Og",
    },
}
LANTHANIDES = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
ACTINIDES = ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]


def periodic_positions():
    positions = {}
    for period, row in PERIOD_BLOCKS.items():
        for group, symbol in row.items():
            positions[symbol] = (group, period)
    for offset, symbol in enumerate(LANTHANIDES):
        positions[symbol] = (3 + offset, 8.8)
    for offset, symbol in enumerate(ACTINIDES):
        positions[symbol] = (3 + offset, 9.8)
    return positions


POSITIONS = periodic_positions()


def count_present(frame):
    """Count formulas containing each element (one count per formula)."""
    missing = set(ELEMENTS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing element columns: {sorted(missing)}")
    present = frame[ELEMENTS].to_numpy(dtype=float) > 0
    return pd.Series(present.sum(axis=0), index=ELEMENTS, dtype=int)


def build_frequency_table(frame):
    counts = count_present(frame)
    n_rows = len(frame)
    table = pd.DataFrame(
        {
            "symbol": SYMBOLS,
            "atomic_number": [ATOMIC_NUMBER[symbol] for symbol in SYMBOLS],
            "count": [int(counts.get(symbol, 0)) for symbol in SYMBOLS],
            "fraction": [round(float(counts.get(symbol, 0)) / n_rows, 6) if n_rows else 0.0 for symbol in SYMBOLS],
        }
    )
    return table.sort_values("atomic_number").reset_index(drop=True)


def draw_periodic(ax, table):
    counts = dict(zip(table["symbol"], table["count"]))
    vmax = max(max(counts.values()), 1)
    cmap = plt.get_cmap("YlGn")
    norm = Normalize(vmin=1, vmax=vmax)

    for symbol, (x, y) in POSITIONS.items():
        count = counts.get(symbol, 0)
        if count:
            red, green, blue, _ = to_rgba(cmap(norm(count)))
            face = (red, green, blue)
            ink = "white" if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.45 else "#222222"
        else:
            face = "#ececec"
            ink = "#222222"
        ax.add_patch(
            Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, facecolor=face,
                      edgecolor="#888888", linewidth=0.25)
        )
        ax.text(x - 0.40, y - 0.38, str(ATOMIC_NUMBER[symbol]),
                ha="left", va="top", fontsize=5.8, color=ink)
        ax.text(x, y - 0.01, symbol, ha="center", va="center",
                fontsize=7.2, color=ink)
        ax.text(x, y + 0.38, f"{count:,}", ha="center", va="bottom",
                fontsize=6.0, color=ink)

    for y, label in ((6, "57-71"), (7, "89-103")):
        ax.add_patch(Rectangle((3 - 0.46, y - 0.46), 0.92, 0.92,
                               facecolor="#ececec", edgecolor="#888888", linewidth=0.25))
        ax.text(3, y, label, ha="center", va="center", fontsize=4.2, color="#555555")

    ax.set_xlim(0.35, 18.65)
    ax.set_ylim(10.25, 0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def plot_frequency(table, outpath):
    plt.rcParams.update({"font.family": "Arial", "font.size": 7})
    fig, ax = plt.subplots(figsize=(18 / 2.54, 11.2 / 2.54))
    fig.subplots_adjust(left=0.02, right=0.90, top=0.97, bottom=0.03)
    sm = draw_periodic(ax, table)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.012, shrink=0.82)
    cbar.set_label("Count", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    cbar.outline.set_linewidth(0.3)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Single periodic-table frequency figure.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Final dataset CSV")
    parser.add_argument("--table", default=str(TABLE_PATH), help="Count CSV")
    parser.add_argument("--figure", default=str(FIGURE_PATH), help="Figure PNG")
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    if "formula" not in frame.columns:
        raise SystemExit("data is missing column: formula")
    table = build_frequency_table(frame)

    table_path = Path(args.table)
    figure_path = Path(args.figure)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    plot_frequency(table, figure_path)

    top = ", ".join(
        f"{row.symbol} {row.count}" for row in table.nlargest(5, "count").itertuples()
    )
    print(f"rows={len(frame):,}; unique formulas={frame['formula'].nunique():,}")
    print(f"elements present={(table['count'] > 0).sum()}/118; top: {top}")
    print(f"saved -> {table_path.relative_to(ROOT)}")
    print(f"saved -> {figure_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
