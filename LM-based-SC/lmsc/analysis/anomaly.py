"""Flag potentially bad rows in the Tc dataset.

Three complementary signals are combined:
  1. out-of-fold residual: fit the selected model on 10 grouped folds, so every
     row is predicted by a model that never saw its exact composition;
  2. duplicate inconsistency: the same normalised composition has a wide Tc
     range, which cannot be explained by composition alone;
  3. composition novelty: IsolationForest on the 81 element fractions finds
     rare/edge compositions (a warning, not proof of a bad label).

Outputs:
  data_anomalies_top.csv   top --top-k rows by suspicion score
  anomaly_summary.json     thresholds, counts, top rows
  results/training/anomaly_*.png  diagnostic plots

Usage:
    python -m lmsc anomaly
    python -m lmsc anomaly --folds 10 --contamination 0.02
    python -m lmsc anomaly --min-abs-error 5 --z-threshold 3.5
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold

from ..estimators import build_model
from ..paths import ROOT, DATASET, TRAINING, public_path

DATA = DATASET
TABLE_DIR = TRAINING
FIG_DIR = TRAINING
METRICS = TABLE_DIR / "metrics.json"
LABEL = "Tc_K"


def load_selected(path=METRICS):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run python -m lmsc train first")
    with open(path, encoding="utf-8") as f:
        metrics = json.load(f)
    if "selected_params" not in metrics or "selected_weight" not in metrics:
        raise KeyError("metrics.json has no selected_params/selected_weight; rerun train.py")
    return metrics


def load_data(path=DATA):
    df = pd.read_csv(path)
    features = [c for c in df.columns if c not in ("formula", LABEL)]
    X, y = df[features], df[LABEL]
    groups = X.round(12).astype(str).agg("|".join, axis=1)
    return df, X, y, groups, features


def fit_fold(X, y, params, alpha, power, seed):
    model = build_model("ExtraTrees", params, seed=seed)
    if alpha:
        weights = (1.0 + alpha * np.asarray(y, dtype=float)) ** power
        weights = weights / weights.mean()
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def oof_predictions(X, y, groups, params, alpha, power, folds, seed):
    oof = np.full(len(y), np.nan, dtype=float)
    fold_id = np.full(len(y), -1, dtype=int)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, test) in enumerate(splitter.split(X, y, groups=groups), start=1):
        model = fit_fold(X.iloc[train], y.iloc[train], params, alpha, power, seed)
        oof[test] = model.predict(X.iloc[test])
        fold_id[test] = fold
        print(f"[oof] fold {fold}/{folds} done", flush=True)
    if np.isnan(oof).any():
        raise RuntimeError("some rows were not predicted out-of-fold")
    return oof, fold_id


def duplicate_stats(groups, y):
    frame = pd.DataFrame({"group": groups.to_numpy(), "y": y.to_numpy(dtype=float)})
    grouped = frame.groupby("group")["y"]
    stats = grouped.agg(
        group_count="size",
        group_mean="mean",
        group_median="median",
        group_std="std",
        group_min="min",
        group_max="max",
    )
    stats["group_range"] = stats["group_max"] - stats["group_min"]
    stats["group_mad"] = grouped.apply(
        lambda s: float(np.median(np.abs(s.to_numpy() - np.median(s.to_numpy()))))
    )
    stats["group_std"] = stats["group_std"].fillna(0.0)
    return stats.reset_index()


def robust_residual(residual, abs_error, z_threshold, min_abs_error):
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    scale = max(1.4826 * mad, 1e-9)
    z = (residual - center) / scale
    flag = (abs_error >= min_abs_error) & (np.abs(z) >= z_threshold)
    return z, flag, center, mad


def add_duplicate_flags(table, duplicate_range, min_abs_error):
    table["group_deviation"] = (table[LABEL] - table["group_median"]).abs()
    table["group_z"] = np.where(
        (table["group_count"] >= 3) & (table["group_mad"] > 1e-9),
        0.6745 * table["group_deviation"] / table["group_mad"].replace(0, np.nan),
        np.nan,
    )
    table["duplicate_flag"] = (
        (table["group_count"] >= 2)
        & (table["group_range"] >= duplicate_range)
        & (table["group_deviation"] >= min_abs_error)
    )
    return table


def reason_text(row):
    reasons = []
    if row["residual_flag"]:
        reasons.append(f"oof_residual={row['abs_error']:.2f}K,z={row['residual_z']:.2f}")
    if row["duplicate_flag"]:
        reasons.append(f"duplicate_range={row['group_range']:.2f}K,n={int(row['group_count'])}")
    if row["isolation_flag"]:
        reasons.append(f"composition_novelty={row['isolation_score']:.4f}")
    return "; ".join(reasons)


def make_plots(table, fig_dir, top_k, cutoff):
    fig_dir.mkdir(parents=True, exist_ok=True)
    suspicious = table["suspicious"].astype(bool)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(table.loc[~suspicious, LABEL], table.loc[~suspicious, "oof_pred"],
               s=8, alpha=0.25, color="#9ecae1", edgecolors="none", label="normal")
    ax.scatter(table.loc[suspicious, LABEL], table.loc[suspicious, "oof_pred"],
               s=22, alpha=0.8, color="#d73027", edgecolors="none", label="suspicious")
    low = min(0.0, float(table[LABEL].min()), float(table["oof_pred"].min())) * 1.02
    high = max(float(table[LABEL].max()), float(table["oof_pred"].max())) * 1.02
    ax.plot([low, high], [low, high], "k--", lw=1, label="y = x")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Observed Tc (K)")
    ax.set_ylabel("Out-of-fold predicted Tc (K)")
    ax.set_title("OOF residuals: suspicious rows highlighted")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "anomaly_residuals.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(table["suspicion_score"], bins=40, color="#4575b4", alpha=0.85)
    ax.axvline(cutoff, color="#d73027", ls="--", lw=1,
               label=f"top {top_k} cutoff")
    ax.set_xlabel("Suspicion score (rank blend)")
    ax.set_ylabel("Rows")
    ax.set_title("Suspicion-score distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "anomaly_score_hist.png", dpi=200)
    plt.close(fig)

    dup = table[table["group_count"] >= 2]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(dup["group_count"], dup["group_range"], s=16, alpha=0.45,
               color="#6a994e", edgecolors="none", label="duplicate group")
    flagged = dup["duplicate_flag"].astype(bool)
    ax.scatter(dup.loc[flagged, "group_count"], dup.loc[flagged, "group_range"],
               s=28, alpha=0.9, color="#d73027", edgecolors="none", label="inconsistent")
    ax.set_xscale("log")
    ax.set_xlabel("Rows in exact composition group (log)")
    ax.set_ylabel("Tc range within group (K)")
    ax.set_title("Duplicate-composition inconsistency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "anomaly_duplicates.png", dpi=200)
    plt.close(fig)

    top = table.nlargest(min(top_k, 20), "suspicion_score").iloc[::-1]
    labels = [
        f"{row['formula']}  (Tc={row[LABEL]:g}, n={int(row['group_count'])})"
        for _, row in top.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.42 * len(top))))
    ax.barh(labels, top["suspicion_score"], color="#d73027", alpha=0.85)
    ax.set_xlabel("Suspicion score")
    ax.set_title(f"Top {len(top)} suspicious rows")
    fig.tight_layout()
    fig.savefig(fig_dir / "anomaly_top.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--top-output", type=Path, default=TABLE_DIR / "data_anomalies_top.csv")
    parser.add_argument("--summary", type=Path, default=TABLE_DIR / "anomaly_summary.json")
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--z-threshold", type=float, default=3.5)
    parser.add_argument("--min-abs-error", type=float, default=5.0)
    parser.add_argument("--duplicate-range", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (args.top_output, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_selected(args.metrics)
    df, X, y, groups, features = load_data(args.data)
    params = metrics["selected_params"]
    alpha = float(metrics["selected_weight"]["alpha"])
    power = float(metrics["selected_weight"]["power"])
    print(f"rows={len(df)}, features={len(features)}, groups={groups.nunique()}")
    print(f"selected model: {json.dumps(params, sort_keys=True)}; weight alpha={alpha}, power={power}")

    oof, fold_id = oof_predictions(X, y, groups, params, alpha, power, args.folds, args.seed)
    residual = oof - y.to_numpy(dtype=float)
    abs_error = np.abs(residual)
    residual_z, residual_flag, residual_center, residual_mad = robust_residual(
        residual, abs_error, args.z_threshold, args.min_abs_error
    )

    table = df.copy()
    table["row_id"] = np.arange(len(table))
    table["oof_pred"] = oof
    table["residual"] = residual
    table["abs_error"] = abs_error
    table["rel_error"] = abs_error / np.maximum(np.abs(y.to_numpy(dtype=float)), 1e-12)
    table["residual_z"] = residual_z
    table["residual_flag"] = residual_flag
    table["oof_fold"] = fold_id
    table["group"] = groups.to_numpy()
    table = table.merge(duplicate_stats(groups, y), on="group", how="left")
    table["group_key"] = table["group"]
    table = add_duplicate_flags(table, args.duplicate_range, args.min_abs_error)

    iso = IsolationForest(
        n_estimators=400,
        contamination=args.contamination,
        random_state=args.seed,
        n_jobs=-1,
    )
    iso.fit(X)
    table["isolation_score"] = -iso.score_samples(X)
    table["isolation_flag"] = iso.predict(X) == -1

    n = len(table)
    abs_rank = rankdata(table["abs_error"], method="average") / n
    dup_rank = rankdata(table["group_range"], method="average") / n
    dup_rank = np.where(table["group_count"] >= 2, dup_rank, 0.0)
    iso_rank = rankdata(table["isolation_score"], method="average") / n
    table["suspicion_score"] = 0.60 * abs_rank + 0.25 * dup_rank + 0.15 * iso_rank
    table["suspicious"] = (
        table["residual_flag"] | table["duplicate_flag"] | table["isolation_flag"]
    )
    table["reasons"] = table.apply(reason_text, axis=1)
    table["severity"] = np.select(
        [
            table["residual_flag"] & (table["abs_error"] >= 2 * args.min_abs_error),
            table["duplicate_flag"] & (table["group_range"] >= 2 * args.duplicate_range),
            table["suspicious"],
        ],
        ["high", "high", "medium"],
        default="low",
    )
    table = table.sort_values(
        ["suspicion_score", "abs_error"], ascending=[False, False]
    ).reset_index(drop=True)
    table["anomaly_rank"] = np.arange(1, len(table) + 1)
    cutoff = float(table.iloc[min(args.top_k, n) - 1]["suspicion_score"])
    table["suspicious_score_threshold"] = cutoff

    columns = [
        "anomaly_rank", "formula", LABEL, "oof_pred", "residual", "abs_error",
        "rel_error", "residual_z", "residual_flag", "oof_fold",
        "group_count", "group_mean", "group_median", "group_std", "group_range",
        "group_deviation", "group_z", "duplicate_flag",
        "isolation_score", "isolation_flag", "suspicion_score",
        "suspicious", "severity", "reasons", "row_id", "group_key",
    ]
    top = table.head(args.top_k)
    top[columns].to_csv(args.top_output, index=False)
    make_plots(table, args.fig_dir, args.top_k, cutoff)

    summary = {
        "data_file": public_path(args.data),
        "n_rows": int(n),
        "n_groups": int(groups.nunique()),
        "n_oof_folds": args.folds,
        "selected_model": params,
        "selected_weight": {"alpha": alpha, "power": power},
        "thresholds": {
            "o_abs_error_K": args.min_abs_error,
            "residual_z": args.z_threshold,
            "duplicate_range_K": args.duplicate_range,
            "isolation_contamination": args.contamination,
            "top_k": args.top_k,
        },
        "residual_stats": {
            "median_residual_K": residual_center,
            "mad_residual_K": residual_mad,
            "oof_rmse_K": float(np.sqrt(np.mean(residual**2))),
            "oof_mae_K": float(np.mean(abs_error)),
        },
        "counts": {
            "residual_flag": int(table["residual_flag"].sum()),
            "duplicate_flag": int(table["duplicate_flag"].sum()),
            "isolation_flag": int(table["isolation_flag"].sum()),
            "suspicious": int(table["suspicious"].sum()),
            "high_severity": int((table["severity"] == "high").sum()),
        },
        "outputs": {
            "top": public_path(args.top_output),
            "figures": [
                public_path(args.fig_dir / "anomaly_residuals.png"),
                public_path(args.fig_dir / "anomaly_score_hist.png"),
                public_path(args.fig_dir / "anomaly_duplicates.png"),
                public_path(args.fig_dir / "anomaly_top.png"),
            ],
        },
        "top_rows": top[
            ["anomaly_rank", "formula", LABEL, "oof_pred", "abs_error",
             "group_count", "group_range", "suspicion_score", "severity", "reasons"]
        ].to_dict("records"),
    }
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== anomaly summary ===")
    print(json.dumps(summary["counts"], indent=2))
    print(f"oof RMSE {summary['residual_stats']['oof_rmse_K']:.4f} K")
    print(f"cutoff suspicion score: {cutoff:.4f}")
    print(top[["anomaly_rank", "formula", LABEL, "oof_pred", "abs_error",
               "group_count", "group_range", "severity", "reasons"]]
          .head(20).to_string(index=False))
    print(f"saved -> {args.top_output}, {args.summary}, {args.fig_dir}/anomaly_*.png")


if __name__ == "__main__":
    main()
