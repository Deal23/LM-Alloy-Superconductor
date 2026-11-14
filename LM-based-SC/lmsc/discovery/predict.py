"""Step 3: predict Tc, search compositions, draw phase diagrams, or analyze results.

Usage:
    python -m lmsc predict In0.5Sn0.5 Nb3Sn
    python -m lmsc predict
    python -m lmsc predict --search --elements In Sn Nb --denominator 12
    python -m lmsc predict --phase-diagram --elements Nb Ga
    python -m lmsc predict --phase-diagram --elements Ga In Ag
    python -m lmsc predict --analyze
"""
import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from ..composition import TARGET, canonical_formula, compositions, feature_row, parse_formula
from ..paths import ROOT, DATASET, MODELS, TRAINING, PREDICTIONS

MODEL_PATH = MODELS / "model.joblib"
COLS_PATH = MODELS / "feature_columns.json"
METRICS_PATH = TRAINING / "metrics.json"
DATA_PATH = DATASET
LOG_PATH = PREDICTIONS / "predict_log.csv"
SEARCH_OUT = PREDICTIONS / "composition_optimization.csv"
FIG_DIR = PREDICTIONS
PHASE_FIG_DIR = FIG_DIR / "phase_diagrams"
PHASE_RESULT_DIR = PHASE_FIG_DIR


def load_artifacts():
    if not MODEL_PATH.exists():
        raise SystemExit("model not found; run python -m lmsc train first")
    model = joblib.load(MODEL_PATH)
    with open(COLS_PATH, encoding="utf-8") as f:
        cols = json.load(f)
    return model, cols


def predict_batch(model, cols, formulas):
    rows = []
    for formula in formulas:
        try:
            frac = parse_formula(formula, allowed=cols)
            X = pd.DataFrame([{c: frac.get(c, 0.0) for c in cols}], columns=cols)
            tc = float(model.predict(X)[0])
            rows.append({"formula": formula, "tc_predicted_K": round(tc, 4)})
        except ValueError as e:
            rows.append({"formula": formula, "tc_predicted_K": np.nan})
            print(f"{formula}: skipped ({e})")
    return pd.DataFrame(rows)


def append_log(df):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = df[["formula", "tc_predicted_K"]].copy()
    new.insert(0, "timestamp", datetime.now().isoformat(timespec="seconds"))
    log = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
    log = pd.concat([log, new], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)


def model_predictions(model, X):
    trees = getattr(model, "estimators_", None)
    candidates = list(trees) if isinstance(trees, (list, np.ndarray)) else []
    if candidates and all(hasattr(tree, "predict") for tree in candidates):
        try:
            preds = np.asarray([tree.predict(X) for tree in candidates])
            return preds.mean(axis=0), preds.std(axis=0)
        except Exception:
            pass
    pred = np.asarray(model.predict(X), dtype=float)
    return pred, np.zeros_like(pred)


def score_chunk(model, columns, pending, denominator):
    if not pending:
        return []
    X = np.asarray([feature_row(elements, counts, columns) for elements, counts in pending])
    mean, std = model_predictions(model, X)
    return [
        {
            "formula": canonical_formula(elements, counts),
            "components": len(elements),
            "denominator": denominator,
            "tc_mean_K": round(float(pred), 5),
            "tc_std_K": round(float(spread), 5),
            "tc_lcb_K": round(float(pred - 1.5 * spread), 5),
        }
        for (elements, counts), pred, spread in zip(pending, mean, std)
    ]


def search(model, columns, elements, component_counts, denominator, chunk_size=100_000, top_k=50):
    elements = list(elements)
    component_counts = tuple(component_counts)
    if chunk_size < 1 or top_k < 1:
        raise ValueError("chunk_size and top_k must be positive")
    if len(set(elements)) != len(elements):
        raise ValueError("elements must be unique")
    unknown = sorted(set(elements) - set(columns))
    if unknown:
        raise ValueError(f"unknown elements: {unknown}")
    if not component_counts or any(n not in (2, 3) or denominator < n for n in component_counts):
        raise ValueError("components must be 2 or 3 and denominator >= components")
    if len(set(component_counts)) != len(component_counts):
        raise ValueError("component counts must be unique")
    top_chunks = []
    candidate_count = 0
    for n_components in component_counts:
        pending = []
        for selected, counts in compositions(elements, n_components, denominator):
            pending.append((selected, counts))
            if len(pending) >= chunk_size:
                chunk = pd.DataFrame(score_chunk(model, columns, pending, denominator))
                top_chunks.append(chunk.nlargest(top_k, "tc_lcb_K"))
                candidate_count += len(chunk)
                pending = []
        chunk = pd.DataFrame(score_chunk(model, columns, pending, denominator))
        if len(chunk):
            top_chunks.append(chunk.nlargest(top_k, "tc_lcb_K"))
            candidate_count += len(chunk)
    if not top_chunks:
        raise ValueError("no candidates: provide enough distinct elements for the requested components")
    result = pd.concat(top_chunks, ignore_index=True).nlargest(top_k, "tc_lcb_K")
    return result, candidate_count


def run_prediction(formulas):
    if not formulas:
        print("enter formula per line (empty line to quit):")
        while True:
            line = input("> ").strip()
            if not line:
                break
            formulas.append(line)
    if not formulas:
        return

    model, cols = load_artifacts()
    out = predict_batch(model, cols, formulas)
    for _, r in out.iterrows():
        if not np.isnan(r["tc_predicted_K"]):
            print(f"{r['formula']}: Tc = {r['tc_predicted_K']:.3f} K")
    if len(out):
        append_log(out)


def run_search(args):
    if not MODEL_PATH.exists():
        raise SystemExit("model not found; run python -m lmsc train first")
    model = joblib.load(MODEL_PATH)
    columns = json.loads(COLS_PATH.read_text(encoding="utf-8"))
    data = pd.read_csv(DATA_PATH)
    support = (data[columns] > 0).sum()
    elements = args.elements or [column for column in columns if support[column] >= args.min_support]
    unknown = sorted(set(elements) - set(columns))
    if unknown:
        raise SystemExit(f"unknown elements: {unknown}")
    components = tuple(int(value) for value in args.components.split(","))
    sizes = [int(value) for value in args.denominators.split(",")] if args.denominators else [args.denominator]

    reports = []
    for denominator in sizes:
        result, candidate_count = search(
            model, columns, elements, components, denominator, args.chunk_size, args.top_k
        )
        report = result.head(args.top_k)
        reports.append(report.assign(candidate_count=candidate_count))
        print(f"D={denominator}: candidates={candidate_count}; best={report.iloc[0]['formula']} "
              f"({report.iloc[0]['tc_mean_K']:.3f} K)")
    output = pd.concat(reports, ignore_index=True)
    output.to_csv(SEARCH_OUT, index=False)
    print(f"saved -> {SEARCH_OUT.relative_to(ROOT)}")


def formula_from_counts(elements, counts):
    total = float(sum(counts))
    parts = []
    for element, count in zip(elements, counts):
        if count <= 0:
            continue
        fraction = count / total
        if np.isclose(fraction, 1.0):
            parts.append(element)
        else:
            parts.append(f"{element}{fraction:.4g}")
    return "".join(parts)


def binary_compositions(elements, denominator):
    if len(elements) != 2 or denominator < 2:
        raise ValueError("binary phase diagram needs 2 elements and denominator >= 2")
    a, b = elements
    for i in range(denominator + 1):
        yield (a, b), (denominator - i, i)


def ternary_compositions(elements, denominator):
    if len(elements) != 3 or denominator < 3:
        raise ValueError("ternary phase diagram needs 3 elements and denominator >= 3")
    a, b, c = elements
    for i in range(denominator + 1):
        for j in range(denominator + 1 - i):
            k = denominator - i - j
            yield (a, b, c), (i, j, k)


def barycentric_xy(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    total = a + b + c
    x = (b + 0.5 * c) / total
    y = (np.sqrt(3.0) / 2.0) * c / total
    return x, y


def select_subsystem(data, columns, elements):
    values = data[columns].to_numpy(dtype=float)
    selected_idx = [columns.index(element) for element in elements]
    selected = values[:, selected_idx].sum(axis=1)
    others = np.delete(values, selected_idx, axis=1)
    mask = np.isclose(selected, 1.0, atol=1e-9) & (np.abs(others).max(axis=1) < 1e-12)
    return data.loc[mask]


def predict_compositions(model, columns, elements, pending):
    counts_list = [counts for _, counts in pending]
    X = np.asarray([feature_row(elements, counts, columns) for counts in counts_list])
    mean, std = model_predictions(model, X)
    rows = []
    for counts, tc_mean, tc_std in zip(counts_list, mean, std):
        total = float(sum(counts))
        row = {"formula": formula_from_counts(elements, counts)}
        for element, count in zip(elements, counts):
            row[f"fraction_{element}"] = round(count / total, 8)
        row.update({
            "tc_mean_K": round(float(tc_mean), 5),
            "tc_std_K": round(float(tc_std), 5),
            "tc_lcb_K": round(float(tc_mean - 1.5 * tc_std), 5),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def plot_binary_phase(frame, elements, observed, output, tag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = elements
    x = frame[f"fraction_{b}"].to_numpy(dtype=float)
    mean = frame["tc_mean_K"].to_numpy(dtype=float)
    std = frame["tc_std_K"].to_numpy(dtype=float)
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.fill_between(x[order], (mean - std)[order], (mean + std)[order],
                    color="#9ecae1", alpha=0.45, label="tree std")
    ax.plot(x[order], mean[order], color="#08519c", lw=2.0, label="predicted mean")
    if len(observed):
        ax.scatter(observed[b], observed[TARGET], s=26, facecolors="none",
                   edgecolors="#d73027", linewidths=0.9, zorder=4,
                   label=f"training data (n={len(observed)})")
    best = frame.loc[frame["tc_mean_K"].idxmax()]
    best_x = float(best[f"fraction_{b}"])
    best_y = float(best["tc_mean_K"])
    ax.scatter([best_x], [best_y], marker="*", s=160, facecolors="white",
               edgecolors="black", linewidths=0.9, zorder=5)
    ax.annotate(f"{best['formula']}\n{best_y:.2f} K", (best_x, best_y),
                xytext=(8, 8), textcoords="offset points", fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.6", "alpha": 0.9})
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(f"x({b}) in {a}–{b}")
    ax.set_ylabel("Predicted Tc (K)")
    ax.set_title(f"Predicted Tc composition map: {a}–{b}")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.text(0.01, 0.01, "Model prediction, not thermodynamic phase boundaries.",
             fontsize=7, color="0.4")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return best


def plot_ternary_phase(frame, elements, observed, output, tag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b, c = elements
    x, y = barycentric_xy(
        frame[f"fraction_{a}"], frame[f"fraction_{b}"], frame[f"fraction_{c}"]
    )
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

    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="black", lw=1.0)
    ax.text(-0.02, -0.03, a, ha="right", va="top", fontsize=10, fontweight="bold")
    ax.text(1.02, -0.03, b, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(0.5, h + 0.03, c, ha="center", va="bottom", fontsize=10, fontweight="bold")

    if len(observed):
        xo, yo = barycentric_xy(observed[a], observed[b], observed[c])
        ax.scatter(xo, yo, s=22, facecolors="white", edgecolors="black",
                   linewidths=0.55, alpha=0.9, zorder=5,
                   label=f"training data (n={len(observed)})")
        ax.legend(frameon=False, fontsize=7, loc="upper left",
                  bbox_to_anchor=(-0.01, 1.0))

    best = frame.loc[frame["tc_mean_K"].idxmax()]
    bx, by = barycentric_xy(
        best[f"fraction_{a}"], best[f"fraction_{b}"], best[f"fraction_{c}"]
    )
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
    fig.text(0.01, 0.01, "Model prediction, not thermodynamic phase boundaries.",
             fontsize=7, color="0.4")
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return best


def run_phase_diagram(args):
    elements = tuple(args.elements or ())
    if len(elements) not in (2, 3):
        raise SystemExit("--phase-diagram needs exactly 2 or 3 elements, e.g. --elements Nb Ga")
    if len(set(elements)) != len(elements):
        raise SystemExit("duplicate elements in --elements")
    model, columns = load_artifacts()
    unknown = sorted(set(elements) - set(columns))
    if unknown:
        raise SystemExit(f"unknown elements: {unknown}")

    grid = args.phase_grid or (100 if len(elements) == 2 else 50)
    if grid < len(elements):
        raise SystemExit("--phase-grid is too small")
    if len(elements) == 2:
        pending = list(binary_compositions(elements, grid))
    else:
        pending = list(ternary_compositions(elements, grid))
    frame = predict_compositions(model, columns, elements, pending)
    data = pd.read_csv(DATA_PATH)
    observed = select_subsystem(data, columns, elements)

    PHASE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "-".join(elements)
    csv_path = PHASE_RESULT_DIR / f"{tag}_phase_compositions.csv"
    png_path = PHASE_FIG_DIR / f"phase_{tag}_diagram.png"
    frame.to_csv(csv_path, index=False)
    if len(observed):
        observed[["formula", TARGET, *elements]].to_csv(
            PHASE_RESULT_DIR / f"{tag}_observed.csv", index=False
        )
    if len(elements) == 2:
        best = plot_binary_phase(frame, elements, observed, png_path, tag)
    else:
        best = plot_ternary_phase(frame, elements, observed, png_path, tag)

    summary = {
        "elements": list(elements),
        "grid_denominator": grid,
        "n_grid_points": int(len(frame)),
        "n_observed_training_rows": int(len(observed)),
        "max_tc_K": round(float(best["tc_mean_K"]), 5),
        "max_formula": best["formula"],
        "mean_tc_K": round(float(frame["tc_mean_K"].mean()), 5),
        "note": "Predicted Tc composition map, not thermodynamic phase boundaries.",
    }
    summary_path = PHASE_RESULT_DIR / f"{tag}_phase_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{tag}: {len(frame)} grid points, {len(observed)} training rows")
    print(f"max predicted Tc: {best['formula']} = {float(best['tc_mean_K']):.3f} K")
    print(f"saved -> {png_path.relative_to(ROOT)}, {csv_path.relative_to(ROOT)}, {summary_path.relative_to(ROOT)}")


def run_analysis():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not LOG_PATH.exists():
        raise SystemExit("no predict log; run prediction first")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LOG_PATH).dropna(subset=["tc_predicted_K"]).drop_duplicates(
        subset=["formula"], keep="last"
    ).sort_values("tc_predicted_K", ascending=False)

    top = df.head(30)[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4.0, 0.28 * len(top))))
    ax.barh(top["formula"], top["tc_predicted_K"], color="#e07b39")
    ax.set_xlabel("Predicted Tc (K)")
    ax.set_title("Top 30 alloys by predicted Tc")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predict_top_tc_ranking.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["tc_predicted_K"], bins=30, color="#3a7ca5", edgecolor="white")
    ax.axvline(5.0, color="red", ls="--", lw=1, label="screening threshold (5 K)")
    ax.set_xlabel("Predicted Tc (K)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of predicted Tc")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predict_tc_distribution.png", dpi=200)
    plt.close(fig)

    summary = {
        "n_unique_formulas": int(len(df)),
        "tc_min": round(float(df["tc_predicted_K"].min()), 3),
        "tc_max": round(float(df["tc_predicted_K"].max()), 3),
        "tc_median": round(float(df["tc_predicted_K"].median()), 3),
        "n_above_5K": int((df["tc_predicted_K"] > 5).sum()),
        "top10": df.head(10).to_dict("records"),
    }
    with open(METRICS_PATH, encoding="utf-8") as f:
        train_metrics = json.load(f)
    print("--- training metrics ---")
    for k in ("rmse", "r2", "accuracy_pct", "mare", "smape_pct"):
        print(f"{k}: {train_metrics.get(k)}")
    print("--- prediction analysis ---")
    print(json.dumps(summary, indent=2))
    print(f"saved -> {FIG_DIR.relative_to(ROOT)}/*.png")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("formulas", nargs="*", help="formulas to predict (default mode)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--search", action="store_true", help="exhaustive binary/ternary search")
    mode.add_argument("--phase-diagram", action="store_true",
                      help="predict Tc over a binary/ternary composition grid and plot it")
    mode.add_argument("--analyze", action="store_true", help="analyze predict_log.csv")
    p.add_argument("--components", default="2,3", help="search: 2, 3, or 2,3")
    p.add_argument("--denominator", type=int, default=20, help="search: integer denominator")
    p.add_argument("--denominators", help="search: comma-separated denominators")
    p.add_argument("--elements", nargs="*", help="search/phase: elements, e.g. Nb Ga or Ga In Ag")
    p.add_argument("--min-support", type=int, default=0, help="search: min training rows per element")
    p.add_argument("--top-k", type=int, default=50, help="search: number of candidates to keep")
    p.add_argument("--chunk-size", type=int, default=100_000, help="search: batch size")
    p.add_argument("--phase-grid", type=int, default=0,
                   help="phase: integer grid denominator (default: 100 binary, 50 ternary)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.search:
        if args.formulas:
            parser.error("formulas cannot be combined with --search")
        run_search(args)
    elif args.phase_diagram:
        if args.formulas:
            parser.error("formulas cannot be combined with --phase-diagram")
        run_phase_diagram(args)
    elif args.analyze:
        if args.formulas:
            parser.error("formulas cannot be combined with --analyze")
        run_analysis()
    else:
        run_prediction(list(args.formulas))


if __name__ == "__main__":
    main()
