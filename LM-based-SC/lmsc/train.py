"""Step 2: compare tree models, tune the best one, and save the final model.

Improvements over the original fixed-grid workflow:
  * cost-sensitive ExtraTrees: sample weights = (1 + Tc) to align training with
    the RMSE-dominated high-Tc tail;
  * Optuna TPE + median pruning instead of a blind full grid;
  * repeated grouped CV for the final model selection, so a single lucky split
    cannot pick the deployment model;
  * one grouped holdout is kept untouched for the final report.

Usage:
    python -m lmsc train
    python -m lmsc train --n-trials 40 --tune-splits 3
    python -m lmsc train --quick
    python -m lmsc train --compare-only
"""
import argparse
import hashlib
import json
import time
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .estimators import DEFAULT_PARAMS, build_model
from .paths import ROOT, DATASET, MODELS, TRAINING

DATA = DATASET
MODEL_DIR = MODELS
TABLE_DIR = TRAINING
LABEL = "Tc_K"
SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.1
REPEAT_SEEDS = (42, 43, 44)
FIG_DIR = TRAINING
WEIGHTED = "ExtraTreesWeighted"
CANDIDATES = [
    ("ExtraTrees", 0.0, 1.0),
    (WEIGHTED, 1.0, 1.0),
    ("RandomForest", 0.0, 1.0),
    ("HistGradientBoosting", 0.0, 1.0),
    ("XGBoost", 0.0, 1.0),
    ("LightGBM", 0.0, 1.0),
    ("CatBoost", 0.0, 1.0),
]
LABELS = {
    "ExtraTrees": "ExtraTrees",
    WEIGHTED: "ExtraTrees + Tc weight",
    "RandomForest": "RandomForest",
    "HistGradientBoosting": "HistGradientBoosting",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "CatBoost": "CatBoost",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = np.abs(err)
    nz = y_true != 0
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "accuracy_pct": float(100 * np.mean(mae < 5)),
        "mare": float(np.mean(mae[nz] / y_true[nz])) if nz.any() else None,
        "smape_pct": float(100 * np.mean(2 * mae / (np.abs(y_true) + np.abs(y_pred) + 1e-12))),
    }


def load_data():
    df = pd.read_csv(DATA)
    assert not df.columns.duplicated().any(), "duplicate columns in data"
    features = [c for c in df.columns if c not in ("formula", LABEL)]
    X, y = df[features], df[LABEL]
    groups = X.round(12).astype(str).agg("|".join, axis=1)
    return df, X, y, groups, features


def fit_candidate(name, params, alpha, power, X, y, seed=SEED):
    base = "ExtraTrees" if name == WEIGHTED else name
    model = build_model(base, params, seed=seed)
    if alpha:
        weights = (1.0 + alpha * np.asarray(y, dtype=float)) ** power
        weights = weights / weights.mean()
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def grouped_splits(X, y, groups, n_splits):
    return [(0, fold, train, test) for fold, (train, test) in enumerate(
        GroupKFold(n_splits=n_splits).split(X, y, groups=groups), start=1
    )]


def repeated_splits(X, y, groups, n_splits, seeds):
    splits = []
    for seed in seeds:
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(X, y, groups=groups), start=1):
            splits.append((seed, fold, train, test))
    return splits


def cv_with_timing(name, params, alpha, power, X, y, groups, splits):
    rows = []
    for repeat, fold, train, test in splits:
        started = time.perf_counter()
        model = fit_candidate(name, params, alpha, power, X.iloc[train], y.iloc[train])
        fit_s = time.perf_counter() - started
        pred = model.predict(X.iloc[test])
        rows.append({
            "model": LABELS.get(name, name),
            "repeat": repeat,
            "fold": fold,
            "fit_seconds": fit_s,
            **evaluate(y.iloc[test], pred),
        })
    return pd.DataFrame(rows)


def summarize_folds(folds):
    summary = {}
    for column in ("rmse", "mae", "r2", "accuracy_pct", "mare", "smape_pct", "fit_seconds"):
        values = pd.to_numeric(folds[column], errors="coerce")
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_std"] = float(values.std(ddof=1))
    summary["n_folds"] = int(len(folds))
    return summary


def compare_models(X, y, groups, n_splits):
    splits = grouped_splits(X, y, groups, n_splits)
    comparisons, all_folds = [], []
    for name, alpha, power in CANDIDATES:
        print(f"[compare] {LABELS[name]} ...", flush=True)
        folds = cv_with_timing(name, None, alpha, power, X, y, groups, splits)
        all_folds.append(folds)
        summary = summarize_folds(folds)
        comparisons.append({
            "model": LABELS[name],
            "weight_alpha": alpha,
            "weight_power": power,
            **summary,
        })
        print(
            f"[compare] {LABELS[name]}: RMSE {summary['rmse_mean']:.4f} "
            f"± {summary['rmse_std']:.4f} K, MAE {summary['mae_mean']:.4f} K",
            flush=True,
        )
    comparison = pd.DataFrame(comparisons).sort_values("rmse_mean").reset_index(drop=True)
    fold_table = pd.concat(all_folds, ignore_index=True)
    comparison.to_csv(TABLE_DIR / "model_comparison.csv", index=False)
    fold_table.to_csv(TABLE_DIR / "model_comparison_folds.csv", index=False)
    return comparison, fold_table


def tune_extra_trees(X, y, groups, n_trials, n_splits, timeout=None, seed=SEED):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splits = grouped_splits(X, y, groups, n_splits)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30, 40, 50]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_float("max_features", 0.08, 1.0),
        }
        alpha = trial.suggest_float("weight_alpha", 0.25, 5.0, log=True)
        power = trial.suggest_float("weight_power", 0.5, 1.5)
        scores = []
        for _, fold, train, test in splits:
            model = fit_candidate(WEIGHTED, params, alpha, power, X.iloc[train], y.iloc[train])
            pred = model.predict(X.iloc[test])
            scores.append(float(np.sqrt(np.mean((pred - y.iloc[test].to_numpy()) ** 2))))
            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=12),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=12, n_warmup_steps=2),
    )
    study.enqueue_trial({
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "weight_alpha": 1.0,
        "weight_power": 1.0,
    })
    print(f"[tune] Optuna TPE + MedianPruner: {n_trials} trials x {n_splits} folds", flush=True)
    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    elapsed = time.perf_counter() - started

    rows = []
    fold_rows = []
    best_so_far = float("inf")
    for trial in study.trials:
        params = dict(trial.params)
        if trial.state.name == "COMPLETE" and trial.value is not None:
            best_so_far = min(best_so_far, float(trial.value))
        rows.append({
            "trial": trial.number,
            "state": trial.state.name,
            "cv_rmse": trial.value,
            "best_so_far": best_so_far if best_so_far < float("inf") else None,
            "duration_s": trial.duration.total_seconds() if trial.duration else None,
            "params_json": json.dumps(params, sort_keys=True, default=str),
            "n_estimators": params.get("n_estimators"),
            "max_depth": params.get("max_depth"),
            "min_samples_split": params.get("min_samples_split"),
            "min_samples_leaf": params.get("min_samples_leaf"),
            "max_features": params.get("max_features"),
            "weight_alpha": params.get("weight_alpha"),
            "weight_power": params.get("weight_power"),
        })
        for step, value in trial.intermediate_values.items():
            fold_rows.append({
                "trial": trial.number,
                "fold": step,
                "partial_cv_rmse": value,
            })
    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "tuning_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(TABLE_DIR / "tuning_fold_scores.csv", index=False)
    complete = results[results["state"] == "COMPLETE"].dropna(subset=["cv_rmse"])
    states = results["state"].value_counts().to_dict()
    print(
        f"[tune] best complete CV RMSE {study.best_value:.4f} K; "
        f"elapsed {elapsed:.1f}s; states={states}",
        flush=True,
    )
    return study, results, elapsed, states


def effective_params(params):
    merged = dict(DEFAULT_PARAMS["ExtraTrees"])
    merged.update(params or {})
    return merged


def candidate_key(params, alpha, power):
    payload = {
        "params": effective_params(params),
        "weight_alpha": float(alpha),
        "weight_power": float(power),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def repeated_cv_finalists(finalists, X, y, groups, n_splits, seeds):
    splits = repeated_splits(X, y, groups, n_splits, seeds)
    rows, summaries = [], []
    for payload in finalists:
        label = payload["label"]
        print(f"[repeated-cv] {label} ...", flush=True)
        folds = cv_with_timing(
            WEIGHTED, payload["params"], payload["weight_alpha"], payload["weight_power"],
            X, y, groups, splits,
        )
        folds["candidate"] = label
        rows.append(folds)
        summary = summarize_folds(folds)
        summary.update({
            "candidate": label,
            "params": payload["params"],
            "weight_alpha": payload["weight_alpha"],
            "weight_power": payload["weight_power"],
        })
        summaries.append(summary)
        print(f"[repeated-cv] {label}: RMSE {summary['rmse_mean']:.4f} K", flush=True)
    table = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame(summaries).sort_values("rmse_mean").reset_index(drop=True)
    summary.to_csv(TABLE_DIR / "repeated_cv_summary.csv", index=False)
    return summary


def choose_robust(summary):
    best = summary.iloc[0]
    threshold = float(best["rmse_mean"] + best["rmse_std"] / np.sqrt(best["n_folds"]))
    eligible = summary[summary["rmse_mean"] <= threshold].copy()
    preferred = eligible[eligible["candidate"] == "ExtraTrees + Tc weight"]
    chosen = preferred.iloc[0] if not preferred.empty else best
    return chosen, threshold


def save_plots(comparison, fold_table, tuning_results, repeated_summary,
               y_holdout, pred_holdout, holdout_metrics):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    view = comparison.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(view))))
    ax.barh(view["model"], view["rmse_mean"], xerr=view["rmse_std"], color="#3a7ca5", alpha=0.85)
    ax.set_xlabel("Grouped 5-fold CV RMSE (K)")
    ax.set_title("Tree-model comparison (mean ± std)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_compare_rmse.png", dpi=200)
    plt.close(fig)

    metrics = [
        ("rmse_mean", "rmse_std", "RMSE (K)"),
        ("mae_mean", "mae_std", "MAE (K)"),
        ("r2_mean", "r2_std", "R²"),
        ("accuracy_pct_mean", "accuracy_pct_std", "5 K accuracy (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, max(6.0, 0.38 * len(view) + 3.0)))
    for ax, (column, error, label) in zip(axes.flat, metrics):
        ax.barh(view["model"], view[column], xerr=view[error], color="#5c8d89", alpha=0.85)
        ax.set_xlabel(label)
        ax.set_title(label)
    fig.suptitle("Model comparison by metric", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_compare_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    pivot = fold_table.pivot_table(index=["repeat", "fold"], columns="model", values="rmse")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for column in pivot.columns:
        ax.plot(range(len(pivot)), pivot[column], marker="o", ms=3, label=column)
    ax.set_xlabel("CV fold")
    ax.set_ylabel("RMSE (K)")
    ax.set_title("Per-fold RMSE")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_compare_folds.png", dpi=200)
    plt.close(fig)

    hist = tuning_results.sort_values("trial")
    complete = hist[hist["state"] == "COMPLETE"].dropna(subset=["cv_rmse"])
    pruned = hist[hist["state"] == "PRUNED"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(complete["trial"], complete["cv_rmse"], s=24, color="#2166ac", label="complete", zorder=3)
    if len(pruned):
        ax.scatter(pruned["trial"], pruned["cv_rmse"], s=24, color="#999999", marker="x", label="pruned", zorder=3)
    ax.plot(hist["trial"], hist["best_so_far"], color="#d6604d", lw=1.8, label="best so far")
    ax.axhline(complete["cv_rmse"].min(), color="gray", ls="--", lw=1, label="best CV RMSE")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("CV RMSE (K)")
    ax.set_title("Hyperparameter search history")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_tune_progress.png", dpi=200)
    plt.close(fig)

    params = ["n_estimators", "max_features", "min_samples_split",
              "min_samples_leaf", "weight_alpha", "weight_power"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    best_row = complete.loc[complete["cv_rmse"].idxmin()]
    for ax, column in zip(axes.flat, params):
        ax.scatter(complete[column], complete["cv_rmse"], s=22, alpha=0.75, color="#3a7ca5")
        ax.scatter([best_row[column]], [best_row["cv_rmse"]], s=90, marker="*",
                   color="#d6604d", zorder=4, label="best trial")
        ax.axhline(best_row["cv_rmse"], color="gray", ls="--", lw=1)
        ax.set_xlabel(column)
        ax.set_ylabel("CV RMSE (K)")
        ax.set_title(column)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Hyperparameter effects on CV RMSE", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_tune_param_effects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    view = repeated_summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.42 * len(view))))
    ax.barh(view["candidate"], view["rmse_mean"], xerr=view["rmse_std"], color="#6a994e", alpha=0.85)
    ax.set_xlabel("Repeated grouped CV RMSE (K)")
    ax.set_title("Robust finalist comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_compare_repeated_cv.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_holdout, pred_holdout, s=14, alpha=0.5, color="#2166ac", edgecolors="none")
    low = min(0.0, float(np.min(y_holdout)), float(np.min(pred_holdout))) * 1.02
    high = max(float(np.max(y_holdout)), float(np.max(pred_holdout))) * 1.02
    ax.plot([low, high], [low, high], "k--", lw=1, label="y = x")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Observed Tc (K)")
    ax.set_ylabel("Predicted Tc (K)")
    ax.set_title(
        f"Holdout: RMSE {holdout_metrics['rmse']:.2f} K, "
        f"R² {holdout_metrics['r2']:.3f}, acc {holdout_metrics['accuracy_pct']:.1f}%"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_holdout_pred_vs_true.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=40, help="Optuna trials")
    parser.add_argument("--tune-splits", type=int, default=3, help="inner CV folds during tuning")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS, help="final repeated CV folds")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="grouped holdout fraction")
    parser.add_argument("--timeout", type=float, default=None, help="Optuna timeout in seconds")
    parser.add_argument("--quick", action="store_true", help="10 trials, 3-fold tuning, 2 repeats")
    parser.add_argument("--compare-only", action="store_true", help="only run model comparison")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    df, X, y, groups, features = load_data()
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    groups_tr = groups.iloc[train_idx]
    print(f"data: {len(df)} rows, {len(features)} features, {groups.nunique()} groups")
    print(f"train: {len(X_tr)} rows / {groups_tr.nunique()} groups; holdout: {len(X_te)} rows")

    comparison, fold_table = compare_models(X_tr, y_tr, groups_tr, args.n_splits)
    best_comparison = comparison.iloc[0]
    print(f"[compare] best model: {best_comparison['model']}")
    if args.compare_only:
        print("compare-only requested; stopping before tuning")
        return

    n_trials = 10 if args.quick else args.n_trials
    tune_splits = min(3, args.n_splits) if args.quick else args.tune_splits
    study, tuning_results, tune_seconds, states = tune_extra_trees(
        X_tr, y_tr, groups_tr, n_trials, tune_splits, args.timeout, seed=seed
    )

    base_raw = {
        "label": "ExtraTrees raw",
        "params": dict(DEFAULT_PARAMS["ExtraTrees"]),
        "weight_alpha": 0.0,
        "weight_power": 1.0,
    }
    weighted_default = {
        "label": "ExtraTrees + Tc weight",
        "params": dict(DEFAULT_PARAMS["ExtraTrees"]),
        "weight_alpha": 1.0,
        "weight_power": 1.0,
    }
    finalists = [base_raw, weighted_default]
    seen = {candidate_key(base_raw["params"], 0.0, 1.0),
            candidate_key(weighted_default["params"], 1.0, 1.0)}
    complete = tuning_results[tuning_results["state"] == "COMPLETE"].dropna(subset=["cv_rmse"])
    for _, row in complete.sort_values("cv_rmse").head(5).iterrows():
        params = json.loads(row["params_json"])
        payload = {
            "label": f"Optuna trial {int(row['trial'])}",
            "params": {k: v for k, v in params.items() if not k.startswith("weight_")},
            "weight_alpha": float(params["weight_alpha"]),
            "weight_power": float(params["weight_power"]),
        }
        key = candidate_key(payload["params"], payload["weight_alpha"], payload["weight_power"])
        if key not in seen:
            seen.add(key)
            finalists.append(payload)
    print(f"[repeated-cv] finalists: {[p['label'] for p in finalists]}", flush=True)
    repeated_summary = repeated_cv_finalists(
        finalists, X_tr, y_tr, groups_tr, args.n_splits, REPEAT_SEEDS if not args.quick else (42, 43)
    )
    selected_row, threshold = choose_robust(repeated_summary)
    selected = next(p for p in finalists if p["label"] == selected_row["candidate"])
    print(f"[select] robust selected: {selected['label']} (1-SE threshold {threshold:.4f} K)")

    baseline = fit_candidate("ExtraTrees", None, 0.0, 1.0, X_tr, y_tr)
    baseline_holdout = evaluate(y_te, baseline.predict(X_te))
    tuned_payload = finalists[2] if len(finalists) > 2 else weighted_default
    tuned = fit_candidate(
        WEIGHTED, tuned_payload["params"], tuned_payload["weight_alpha"],
        tuned_payload["weight_power"], X_tr, y_tr,
    )
    tuned_holdout = evaluate(y_te, tuned.predict(X_te))
    selected_model = fit_candidate(
        WEIGHTED, selected["params"], selected["weight_alpha"],
        selected["weight_power"], X_tr, y_tr,
    )
    selected_holdout = evaluate(y_te, selected_model.predict(X_te))

    fe_se_mask = (
        np.isclose(X_te["Fe"].to_numpy(dtype=float), 0.5)
        & np.isclose(X_te["Se"].to_numpy(dtype=float), 0.5)
    )
    baseline_no_fe = evaluate(y_te[~fe_se_mask], baseline.predict(X_te[~fe_se_mask]))
    tuned_no_fe = evaluate(y_te[~fe_se_mask], tuned.predict(X_te[~fe_se_mask]))
    selected_no_fe = evaluate(y_te[~fe_se_mask], selected_model.predict(X_te[~fe_se_mask]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(selected_model, model_path, compress=3)
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_file": DATA.relative_to(ROOT).as_posix(),
        "data_sha256": sha256(DATA),
        "n_rows": int(len(df)),
        "n_train": int(len(X_tr)),
        "n_holdout": int(len(X_te)),
        "n_features": len(features),
        "n_groups": int(groups.nunique()),
        "random_state": seed,
        "split": "grouped_exact_normalized_composition",
        "holdout_size": args.test_size,
        "n_splits": args.n_splits,
        "models_compared": list(comparison["model"]),
        "best_comparison": best_comparison.to_dict(),
        "tuning": {
            "method": "Optuna TPE + MedianPruner",
            "n_trials": n_trials,
            "tune_splits": tune_splits,
            "best_cv_rmse_K": float(study.best_value),
            "elapsed_s": round(tune_seconds, 2),
            "states": states,
            "best_params": study.best_params,
        },
        "repeated_cv": repeated_summary.to_dict("records"),
        "selection": {
            "rule": "repeated grouped CV; one-SE; prefer pre-specified ExtraTrees+Tc weight",
            "threshold_K": threshold,
            "selected_label": selected["label"],
            "selected_repeated_cv": selected_row.to_dict(),
        },
        "selected_params": selected["params"],
        "selected_weight": {
            "alpha": selected["weight_alpha"],
            "power": selected["weight_power"],
        },
        "model_file_mb": round(model_path.stat().st_size / 1e6, 2),
        "baseline_holdout": baseline_holdout,
        "tuned_holdout": tuned_holdout,
        "selected_holdout": selected_holdout,
        "holdout_no_Fe1Se1": {
            "n_rows": int((~fe_se_mask).sum()),
            "baseline": baseline_no_fe,
            "tuned": tuned_no_fe,
            "selected": selected_no_fe,
        },
        "fit_time_s": round(float(selected_row["fit_seconds_mean"]), 2),
        **selected_holdout,
    }
    with open(TABLE_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)

    save_plots(
        comparison, fold_table, tuning_results, repeated_summary,
        y_te, selected_model.predict(X_te), selected_holdout,
    )

    print("\n=== summary ===")
    print(comparison[["model", "rmse_mean", "rmse_std", "mae_mean", "accuracy_pct_mean"]].to_string(index=False))
    print(f"\nrobust finalists:\n{repeated_summary[['candidate', 'rmse_mean', 'rmse_std', 'mae_mean']].to_string(index=False)}")
    print(f"selected: {selected['label']}")
    print(f"params:   {json.dumps(selected['params'], sort_keys=True)}")
    print(f"weight:   alpha={selected['weight_alpha']:.4f}, power={selected['weight_power']:.4f}")
    print(f"size:     {record['model_file_mb']:.1f} MB")
    print(f"baseline holdout: {json.dumps(baseline_holdout, indent=2)}")
    print(f"tuned holdout:    {json.dumps(tuned_holdout, indent=2)}")
    print(f"selected holdout: {json.dumps(selected_holdout, indent=2)}")
    print(f"selected holdout without Fe1Se1: {json.dumps(selected_no_fe, indent=2)}")
    print("saved -> models/model.joblib, models/feature_columns.json, results/training/")


if __name__ == "__main__":
    main()
