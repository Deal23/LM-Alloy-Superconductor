"""Scan every ternary system anchored on a base element and rank by peak predicted Tc.

For each base element B, enumerate all B-X-Y combinations over the model feature
columns, evaluate Tc on a coarse ternary grid, rank by the maximum predicted Tc
inside a genuine-ternary region (all three fractions >= --min-fraction), then
redraw the top-N systems on a fine grid with mean/std/lower-confidence columns.

Ranking anywhere on the triangle is useless here: ExtraTrees is piecewise constant,
so the peak always sits on a binary edge with the base element diluted to ~4%, and
all four base elements return the same Fe-Se / Fe-Yb / B-Mg partners. The
--min-fraction constraint forces the base element to be a real constituent.

Usage:
    python -m lmsc scan
    python -m lmsc scan --bases Ga Bi --top 5
    python -m lmsc scan --min-fraction 0.33

Outputs (separate from the existing phase_diagrams files):
    results/predictions/phase_diagrams/base_scan/<B>/phase_<B>-<X>-<Y>_diagram.png
    results/predictions/phase_diagrams/base_scan/base_scan_overview.png
    results/predictions/phase_diagrams/base_scan/*.csv
"""
import argparse
import itertools
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from ..composition import TARGET
from .predict import barycentric_xy, model_predictions, select_subsystem
from ..paths import ROOT, DATASET, MODELS, PREDICTIONS

CALIBRATION = [
    ("Nb3Sn", 18.0, "well anchored"),
    ("V3Ga", 16.5, "well anchored"),
    ("Nb3Ga", 20.3, "slightly under"),
    ("Nb3Al", 19.0, "well anchored"),
    ("Nb3Ge", 23.0, "under-predicted"),
    ("MgB2", 39.0, "well anchored"),
    ("BiIn2", 5.6, "well anchored"),
    ("BiIn", 4.0, "over-predicted"),
    ("PbIn", 6.6, "well anchored"),
    ("FeSe", 8.0, "artifact: bulk 8 K, dataset has 60-80 K layer/high-pressure rows"),
    ("YbFe2", 0.0, "artifact: not superconducting, model returns 20+ K"),
]

MODEL_PATH = MODELS / "model.joblib"
COLS_PATH = MODELS / "feature_columns.json"
DATA_PATH = DATASET

FIG_DIR = PREDICTIONS / "phase_diagrams" / "base_scan"
TAB_DIR = FIG_DIR


def write_calibration(model, columns, path):
    from ..composition import parse_formula

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for formula, literature, verdict in CALIBRATION:
            fractions = parse_formula(formula, allowed=columns)
            X = pd.DataFrame([{c: fractions.get(c, 0.0) for c in columns}], columns=columns)
            predicted = float(model.predict(X)[0])
            rows.append({
                "formula": formula,
                "literature_Tc_K": literature,
                "predicted_Tc_K": round(predicted, 3),
                "error_K": round(predicted - literature, 3),
                "verdict": verdict,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def ternary_grid(denominator):
    """(n,3) array of fractions ordered as (base, second, third); base varies slowest."""
    rows = []
    for i in range(denominator + 1):
        for j in range(denominator + 1 - i):
            k = denominator - i - j
            rows.append((i / denominator, j / denominator, k / denominator))
    return np.asarray(rows, dtype=float)


def formula_from_fractions(elements, fractions, digits=3):
    parts = []
    for element, value in zip(elements, fractions):
        if value <= 0:
            continue
        parts.append(element if np.isclose(value, 1.0) else f"{element}{value:.{digits}g}")
    return "".join(parts)


def grid_matrix(elements, column_index, n_columns, counts, denominator):
    X = np.zeros((len(counts), n_columns), dtype=float)
    idx = [column_index[e] for e in elements]
    for n, counts_row in enumerate(counts):
        X[n, idx] = np.asarray(counts_row, dtype=float) / float(denominator)
    return X


def screen_base(model, columns, base, partners, denominator, min_fraction, chunk=400):
    """Return one row per base-X-Y system, with peaks inside and outside the constraint."""
    grid = ternary_grid(denominator)
    n_points = len(grid)
    interior = np.all(grid > 0, axis=1)
    constrained = np.all(grid >= min_fraction - 1e-9, axis=1)
    if not constrained.any():
        raise SystemExit(
            f"--min-fraction {min_fraction} leaves no grid point at denominator {denominator}; "
            f"use at least {int(np.ceil(min_fraction * denominator))}/{denominator} or raise --screen-grid"
        )
    col_index = {name: i for i, name in enumerate(columns)}
    base_idx = col_index[base]

    combos = list(itertools.combinations(partners, 2))
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start in range(0, len(combos), chunk):
            block = combos[start:start + chunk]
            X = np.zeros((len(block) * n_points, len(columns)), dtype=float)
            for offset, (first, second) in enumerate(block):
                sl = slice(offset * n_points, (offset + 1) * n_points)
                X[sl, [base_idx, col_index[first], col_index[second]]] = grid
            pred = model.predict(X).reshape(len(block), n_points)
            for offset, (first, second) in enumerate(block):
                values = pred[offset]
                elements = (base, first, second)
                peak_any = int(np.argmax(values))
                peak_con = int(np.argmax(np.where(constrained, values, -np.inf)))
                peak_int = int(np.argmax(np.where(interior, values, -np.inf)))
                records.append({
                    "base": base,
                    "partner_a": first,
                    "partner_b": second,
                    "max_tc_free_K": round(float(values[peak_any]), 4),
                    "max_formula_free": formula_from_fractions(elements, grid[peak_any]),
                    "max_tc_interior_K": round(float(values[peak_int]), 4),
                    "max_formula_interior": formula_from_fractions(elements, grid[peak_int]),
                    "max_tc_K": round(float(values[peak_con]), 4),
                    "max_formula": formula_from_fractions(elements, grid[peak_con]),
                    "constraint_bound_fraction": round(float(grid[peak_con].min()), 4),
                    "mean_tc_K": round(float(values.mean()), 4),
                    "n_points_above_5K": int((values > 5).sum()),
                    "n_points_above_10K": int((values > 10).sum()),
                })
    return pd.DataFrame(records)


def plot_marked_ternary(frame, elements, observed, output, min_fraction, note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b, c = elements
    x, y = barycentric_xy(frame[f"fraction_{a}"], frame[f"fraction_{b}"], frame[f"fraction_{c}"])
    z = frame["tc_mean_K"].to_numpy(dtype=float)
    h = np.sqrt(3.0) / 2.0
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    contour = ax.tricontourf(x, y, z, levels=24, cmap="turbo")
    ax.tricontour(x, y, z, levels=8, colors="white", linewidths=0.3, alpha=0.45)
    fig.colorbar(contour, ax=ax, shrink=0.78, pad=0.02, label="Predicted Tc (K)")

    for fraction in (0.2, 0.4, 0.6, 0.8):
        ax.plot([1 - fraction, 0.5 * (1 - fraction)], [0, h * (1 - fraction)],
                color="0.25", lw=0.35, alpha=0.45)
        ax.plot([fraction, fraction + 0.5 * (1 - fraction)], [0, h * (1 - fraction)],
                color="0.25", lw=0.35, alpha=0.45)
        ax.plot([0.5 * fraction, 1 - 0.5 * fraction], [h * fraction, h * fraction],
                color="0.25", lw=0.35, alpha=0.45)

    if min_fraction > 0:
        inner = [(min_fraction, min_fraction, 1 - 2 * min_fraction),
                 (min_fraction, 1 - 2 * min_fraction, min_fraction),
                 (1 - 2 * min_fraction, min_fraction, min_fraction),
                 (min_fraction, min_fraction, 1 - 2 * min_fraction)]
        px, py = barycentric_xy(*[[p[0] for p in inner], [p[1] for p in inner], [p[2] for p in inner]])
        ax.plot(px, py, color="0.15", lw=0.9, ls=(0, (4, 2)), alpha=0.75, zorder=4)
        ax.text(px[0], max(py) + 0.012, f"inner region: all >= {min_fraction:.0%}", fontsize=7,
                color="0.15", ha="center")

    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="black", lw=1.0)
    ax.text(-0.02, -0.03, a, ha="right", va="top", fontsize=10, fontweight="bold")
    ax.text(1.02, -0.03, b, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(0.5, h + 0.03, c, ha="center", va="bottom", fontsize=10, fontweight="bold")

    if len(observed):
        xo, yo = barycentric_xy(observed[a], observed[b], observed[c])
        ax.scatter(xo, yo, s=22, facecolors="white", edgecolors="black",
                   linewidths=0.55, alpha=0.9, zorder=5,
                   label=f"training data (n={len(observed)})")
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(-0.01, 1.0))

    region = frame[np.all(frame[[f"fraction_{a}", f"fraction_{b}", f"fraction_{c}"]].to_numpy()
                           >= min_fraction - 1e-9, axis=1)]
    best = region.loc[region["tc_mean_K"].idxmax()]
    bx, by = barycentric_xy(best[f"fraction_{a}"], best[f"fraction_{b}"], best[f"fraction_{c}"])
    ax.scatter([bx], [by], marker="*", s=170, facecolors="white",
               edgecolors="black", linewidths=0.9, zorder=6)
    ax.annotate(f"{best['formula']}\n{float(best['tc_mean_K']):.2f} K",
                (bx, by), xytext=(10, 10), textcoords="offset points", fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.6", "alpha": 0.9})
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.06, h + 0.08)
    ax.axis("off")
    ax.set_title(f"Predicted Tc composition map: {a}–{b}–{c}")
    fig.text(0.01, 0.01, note, fontsize=7, color="0.4")
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return best


def build_overview(selected, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    bases = list(dict.fromkeys(selected["base"].tolist()))
    per_row = int(selected.groupby("base").size().max())
    h = np.sqrt(3.0) / 2.0
    fig, axes = plt.subplots(len(bases), per_row, figsize=(3.6 * per_row, 3.5 * len(bases)))
    axes = np.atleast_2d(axes)

    for row, base in enumerate(bases):
        subset = selected[selected["base"] == base].reset_index(drop=True)
        frames = [pd.read_csv(path) for path in subset["grid_csv"]]
        vmin = min(float(f["tc_mean_K"].min()) for f in frames)
        vmax = max(float(f["tc_mean_K"].max()) for f in frames)
        norm = Normalize(vmin=vmin, vmax=vmax)
        for col in range(per_row):
            ax = axes[row, col]
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.10, h + 0.14)
            if col >= len(subset):
                continue
            info = subset.iloc[col]
            frame = frames[col]
            a, b, c = info["elements"].split("-")
            x, y = barycentric_xy(frame[f"fraction_{a}"], frame[f"fraction_{b}"], frame[f"fraction_{c}"])
            ax.tricontourf(x, y, frame["tc_mean_K"].to_numpy(dtype=float), levels=20, cmap="turbo", norm=norm)
            ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="black", lw=0.7)
            pool = frame[frame["in_ternary_region"]] if "in_ternary_region" in frame else frame
            best = pool.loc[pool["tc_mean_K"].idxmax()]
            bx, by = barycentric_xy(best[f"fraction_{a}"], best[f"fraction_{b}"], best[f"fraction_{c}"])
            ax.scatter([bx], [by], marker="*", s=90, facecolors="white",
                       edgecolors="black", linewidths=0.8, zorder=6)
            ax.text(-0.02, -0.05, a, ha="right", va="top", fontsize=8, fontweight="bold")
            ax.text(1.02, -0.05, b, ha="left", va="top", fontsize=8, fontweight="bold")
            ax.text(0.5, h + 0.02, c, ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_title(f"#{int(info['rank'])}  {a}–{b}–{c}\npeak {float(info['peak_tc_K']):.2f} K, "
                         f"std {float(info['peak_tc_std_K']):.2f} K, n={int(info['n_training_rows'])}",
                         fontsize=8)
        mappable = ScalarMappable(norm=norm, cmap="turbo")
        fig.colorbar(mappable, ax=axes[row, :].tolist(), shrink=0.85, pad=0.01,
                     label=f"{base}-based Tc (K)")

    fig.suptitle("Top 5 ternary subsystems per base element — peak model-predicted Tc\n"
                 "Model prediction, not thermodynamic phase boundaries.", fontsize=12)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", nargs="*", default=["Ga", "Bi", "In", "Sn"])
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--screen-grid", type=int, default=50,
                        help="grid denominator used for both screening and the final figures")
    parser.add_argument("--final-grid", type=int, default=50)
    parser.add_argument("--min-fraction", type=float, default=0.2,
                        help="every one of the three elements must reach this fraction")
    parser.add_argument("--min-support", type=int, default=0,
                        help="drop partner elements with fewer than N training rows")
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    columns = json.loads(COLS_PATH.read_text(encoding="utf-8"))
    data = pd.read_csv(DATA_PATH)
    support = {c: int((data[c] > 0).sum()) for c in columns}
    col_index = {name: i for i, name in enumerate(columns)}

    screens = []
    for base in args.bases:
        partners = [c for c in columns if c != base and support[c] >= args.min_support]
        frame = screen_base(model, columns, base, partners, args.screen_grid, args.min_fraction)
        frame["rank"] = frame["max_tc_K"].rank(ascending=False, method="min").astype(int)
        frame["rank_free"] = frame["max_tc_free_K"].rank(ascending=False, method="min").astype(int)
        frame["support_partner_a"] = frame["partner_a"].map(support)
        frame["support_partner_b"] = frame["partner_b"].map(support)
        screens.append(frame)
        print(f"{base}: {len(partners)} partners -> {len(frame)} systems screened "
              f"(peak inside all >= {args.min_fraction:.0%}: {frame['max_tc_K'].max():.2f} K)")

    screen = pd.concat(screens, ignore_index=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    screen.sort_values(["base", "rank"]).to_csv(TAB_DIR / "base_scan_all_systems.csv", index=False)
    calibration = write_calibration(model, columns, TAB_DIR / "base_scan_calibration.csv")
    print("calibration (literature vs model):")
    print(calibration.to_string(index=False))

    selected_rows = []
    for base in args.bases:
        subset = screen[screen["base"] == base].nlargest(args.top, "max_tc_K").reset_index(drop=True)
        for rank, row in subset.iterrows():
            elements = (base, row["partner_a"], row["partner_b"])
            observed = select_subsystem(data, columns, elements)
            counts = [(i, j, args.final_grid - i - j)
                      for i in range(args.final_grid + 1)
                      for j in range(args.final_grid + 1 - i)]
            X = grid_matrix(elements, col_index, len(columns), counts, args.final_grid)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean, std = model_predictions(model, X)
            fractions = np.asarray(counts, dtype=float) / args.final_grid
            frame = pd.DataFrame({
                "formula": [formula_from_fractions(elements, f) for f in fractions],
                **{f"fraction_{e}": fractions[:, n] for n, e in enumerate(elements)},
                "tc_mean_K": np.round(mean, 5),
                "tc_std_K": np.round(std, 5),
                "tc_lcb_K": np.round(mean - 1.5 * std, 5),
            })
            mask = np.all(fractions >= args.min_fraction - 1e-9, axis=1)
            frame["in_ternary_region"] = mask
            tag = f"{base}-{row['partner_a']}-{row['partner_b']}"
            base_dir = TAB_DIR / base
            base_dir.mkdir(parents=True, exist_ok=True)
            csv_path = base_dir / f"{tag}_phase_compositions.csv"
            png_path = base_dir / f"phase_{tag}_diagram.png"
            frame.to_csv(csv_path, index=False)
            note = ("Model prediction, not thermodynamic phase boundaries. Star/dashed triangle = peak and "
                    f"boundary of the query region where every element reaches {args.min_fraction:.0%}.")
            peak = frame[mask].loc[frame[mask]["tc_mean_K"].idxmax()]
            plot_marked_ternary(frame, elements, observed, png_path, args.min_fraction, note)
            selected_rows.append({
                "base": base,
                "rank": rank + 1,
                "elements": tag,
                "partner_a": row["partner_a"],
                "partner_b": row["partner_b"],
                "peak_tc_K": round(float(peak["tc_mean_K"]), 4),
                "peak_formula": peak["formula"],
                "peak_tc_std_K": round(float(peak["tc_std_K"]), 4),
                "peak_tc_lcb_K": round(float(peak["tc_lcb_K"]), 4),
                "fraction_base_at_peak": round(float(peak[f"fraction_{base}"]), 3),
                "peak_on_constraint_boundary": bool(
                    np.isclose(float(peak[[f"fraction_{e}" for e in elements]].min()), args.min_fraction)
                ),
                "mean_tc_K": round(float(frame["tc_mean_K"].mean()), 4),
                "n_points_above_5K": int((frame["tc_mean_K"] > 5).sum()),
                "n_training_rows": int(len(observed)),
                "support_partner_a": int(row["support_partner_a"]),
                "support_partner_b": int(row["support_partner_b"]),
                "peak_tc_free_K": float(row["max_tc_free_K"]),
                "peak_formula_free": row["max_formula_free"],
                "grid_csv": str(csv_path),
                "figure": png_path.relative_to(ROOT).as_posix(),
            })
            print(f"[{base} #{rank + 1}] {tag}: {float(peak['tc_mean_K']):.2f} K @ {peak['formula']} "
                  f"(base {float(peak[f'fraction_{base}']):.0%}, {len(observed)} training rows)")

    selected = pd.DataFrame(selected_rows)
    selected.drop(columns=["grid_csv"]).to_csv(TAB_DIR / "base_scan_top_systems.csv", index=False)
    build_overview(selected, FIG_DIR / "base_scan_overview.png")
    print(f"saved -> {TAB_DIR.relative_to(ROOT)}")
    print(f"saved -> {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
