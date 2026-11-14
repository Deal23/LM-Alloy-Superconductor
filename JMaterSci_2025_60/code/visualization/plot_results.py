"""Regenerate compact data, tuning, sampling, prediction and ternary figures."""
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, FIGURES, ROOT, parse_formula


def save(fig, name, output):
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output / f"{name.lower().replace('-', '_')}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_data(output):
    frame = pd.read_csv(DATA / "processed/mdr_clean.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(frame.Tc, bins=30, color="#427b69")
    axes[0].set(xlabel="Tc (K)", ylabel="Records")
    counts = frame.formula.map(lambda x: len(parse_formula(x))).value_counts().sort_index()
    axes[1].bar(counts.index, counts.values, color="#427b69")
    axes[1].set(xlabel="Number of elements", ylabel="Records")
    save(fig, "data_distributions", output)
    counts = {}
    for formula in frame.loc[frame.Tc > 5, "formula"]:
        for element in parse_formula(formula):
            counts[element] = counts.get(element, 0) + 1
    values = pd.Series(counts).sort_values(ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(values.index, values.values)
    ax.set(ylabel="Records (Tc > 5 K)", title="Top 30 element frequencies")
    save(fig, "data_element_frequency", output)


def plot_tuning(output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model in ["et", "rf"]:
        frame = pd.read_csv(ROOT / f"data/research/tuning/hyperparameters_{model}.csv")
        axes[0].plot(np.sqrt(np.maximum(frame["mse"], 0)), label=model.upper(), alpha=0.7)
        axes[1].scatter(frame["fit_time"], np.sqrt(np.maximum(frame["mse"], 0)), label=model.upper(), s=10)
    axes[0].set(xlabel="Saved grid row", ylabel="Historical holdout RMSE (K)")
    axes[1].set(xlabel="Fit time (s)", ylabel="Historical holdout RMSE (K)")
    axes[0].legend(); axes[1].legend()
    save(fig, "model_tune_history", output)


def plot_prediction(output):
    for model in ["ET", "RF"]:
        frame = pd.read_csv(ROOT / f"data/research/prediction/updated_mdr_{model.lower()}.csv")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].scatter(frame.Tc, frame.Predicted, s=3, alpha=0.4)
        axes[0].plot([0, 150], [0, 150], "k--")
        axes[0].set(xlabel="Reported Tc (K)", ylabel="Predicted Tc (K)")
        axes[1].scatter(frame.Tc, frame.Predicted - frame.Tc, s=3, alpha=0.4)
        axes[1].axhline(0, color="black", ls="--")
        axes[1].set(xlabel="Reported Tc (K)", ylabel="Residual (K)")
        fig.suptitle(f"{model}: historical full-data prediction (not independent test)")
        save(fig, f"predict_{model}_residuals", output)


def plot_sampling(output):
    fig, ax = plt.subplots(figsize=(7, 4))
    sizes = [(50, "50"), (200, "200"), (500, "500"), (1000, "1k"), (2000, "2k"), (5000, "5k")]
    for rank in range(3):
        values = [pd.read_csv(ROOT / f"data/research/sampling/alloy_1_to_3_{suffix}.csv").Predicted_Tc.nlargest(3).iloc[rank]
                  for _, suffix in sizes]
        ax.plot([n for n, _ in sizes], values, "o-", label=f"Rank {rank + 1}")
    ax.set(xlabel="Samples per alloy system", ylabel="Top predicted Tc (K)", xscale="log")
    ax.legend()
    save(fig, "screening_sampling_convergence", output)


def plot_ternary(output):
    for model in ["ET", "RF"]:
        frame = pd.read_csv(ROOT / f"data/research/ga_in_sn/ga_in_sn_{model.lower()}.csv")
        values = pd.DataFrame([parse_formula(x) for x in frame.formula]).fillna(0)
        x = values["Sn"] + 0.5 * values["Ga"]
        y = np.sqrt(3) / 2 * values["Ga"]
        fig, ax = plt.subplots(figsize=(6, 5))
        points = ax.scatter(x, y, c=frame.Predicted_Tc, s=5, cmap="cividis")
        ax.plot([0, 1, .5, 0], [0, 0, np.sqrt(3)/2, 0], color="black", lw=.7)
        ax.text(.5, .89, "Ga", ha="center"); ax.text(0, -.04, "In"); ax.text(1, -.04, "Sn")
        ax.set_aspect("equal"); ax.axis("off")
        fig.colorbar(points, ax=ax, label="Predicted Tc (K)")
        save(fig, f"phase_Ga-In-Sn_{model}", output)


def plot_experiments(output):
    for path in sorted((ROOT / "data/research/experiments").glob("*.csv")):
        frame = pd.read_csv(path)
        columns = [c for c in frame if c.startswith(("ET_", "ET-", "RF-"))]
        x = np.arange(len(frame))
        fig, ax = plt.subplots(figsize=(9, 4))
        width = .8 / len(columns)
        for index, column in enumerate(columns):
            ax.bar(x + (index - (len(columns)-1)/2) * width,
                   frame[column] - frame["Tc" if "Tc" in frame else "Tc (K)"],
                   width=width, label=column)
        labels = frame["formula"] if "formula" in frame else frame.iloc[:, 0]
        ax.set_xticks(x, labels, rotation=15, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=.7)
        ax.set_ylabel("Predicted minus experimental Tc (K)")
        ax.legend(fontsize=8)
        save(fig, f"experiment_{path.stem}", output)


def plot_screening(output):
    for model in ["et", "rf"]:
        binary = pd.read_csv(ROOT / f"data/research/screening/alloy_2_{model}.csv")
        elements = sorted(set("-".join(binary.elements).split("-")))
        indices = {element: index for index, element in enumerate(elements)}
        matrix = np.full((len(elements), len(elements)), np.nan)
        for row in binary.itertuples():
            a, b = (indices[e] for e in row.elements.split("-"))
            matrix[a, b] = matrix[b, a] = row.Predicted_Tc
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(matrix, cmap="cividis")
        ticks = list(range(0, len(elements), 5))
        ax.set_xticks(ticks, [elements[i] for i in ticks], rotation=90)
        ax.set_yticks(ticks, [elements[i] for i in ticks])
        fig.colorbar(image, ax=ax, label="Maximum predicted Tc (K)")
        ax.set_title(f"{model.upper()} binary screening")
        save(fig, f"screening_binary_{model.upper()}", output)

        ternary = pd.read_csv(ROOT / f"data/research/screening/alloy_3_{model}.csv")
        points, values = [], []
        for row in ternary.itertuples():
            for triple in itertools.permutations(indices[e] for e in row.elements.split("-")):
                if triple[0] <= 30 or triple[1] <= 30 or triple[2] <= 20:
                    points.append(triple)
                    values.append(row.Predicted_Tc)
        points = np.asarray(points)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection="3d")
        image = ax.scatter(*points.T, c=values, s=1, cmap="cividis", alpha=.6)
        ax.set(xlabel="Element index", ylabel="Element index", zlabel="Element index",
               title=f"{model.upper()} ternary screening (not composition coordinates)")
        fig.colorbar(image, ax=ax, shrink=.6, label="Maximum predicted Tc (K)")
        save(fig, f"screening_ternary_{model.upper()}", output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["all", "data", "tuning", "prediction", "sampling", "ternary", "experiments", "screening"], default="all")
    p.add_argument("--output", type=Path, default=FIGURES)
    args = p.parse_args()
    for name, function in [("data", plot_data), ("tuning", plot_tuning), ("prediction", plot_prediction),
                           ("sampling", plot_sampling), ("ternary", plot_ternary),
                           ("experiments", plot_experiments), ("screening", plot_screening)]:
        if args.kind in ("all", name):
            function(args.output)


if __name__ == "__main__":
    main()
