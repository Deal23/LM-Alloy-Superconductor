import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
import os

from ml.core.config import PROJECT_ROOT, DATA_DIR
from utils import ELEMENTS

color2 = "#858F03"
color1 = "#115F38"


def plot_fig2a(dataframe=None):
    file_path = os.path.join(DATA_DIR, "superconductors.csv")
    df = dataframe if dataframe is not None else pd.read_csv(file_path)
    data = df["Tc_K"].replace([np.inf, -np.inf], np.nan).dropna().tolist()

    fig_height = 7
    fig_width = 9
    fig, ax1 = plt.subplots(figsize=(fig_width / 2.54, fig_height / 2.54))
    plt.rcParams["font.family"] = "Arial"
    ax1.hist(data, bins=30, color=color1, alpha=0.6, edgecolor="black")
    ax1.set_xlabel(r"$\mathit{T}_{\mathit{c}}$ (K)", fontsize=10)
    ax1.set_ylabel("Number of Formulas", color="black", size=10)
    ax1.tick_params(axis="y", labelcolor=color1, labelsize=8)

    ax2 = ax1.twinx()
    if len(data) > 1 and np.std(data) > 0:
        from scipy.stats import gaussian_kde
        x = np.linspace(min(data), max(data), 200)
        ax2.plot(x, gaussian_kde(data)(x), color=color2)
    ax2.set_ylabel("Density", color="black", size=10)
    ax2.tick_params(axis="y", labelcolor=color2, labelsize=8)

    ax1.tick_params(axis="x", labelsize=8)
    # Let Matplotlib derive both axes from the histogram and density data.
    # A fixed 150 K bound clipped high-Tc datasets.
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()

    plt.tight_layout()
    plt.show()


def plot_fig2b(dataframe=None):
    file_path = os.path.join(DATA_DIR, "superconductors.csv")
    df = dataframe if dataframe is not None else pd.read_csv(file_path)

    def count_elements(formula):
        from utils.scope import parse_scoped_formula
        return len(parse_scoped_formula(formula, allow_zero=True, unrestricted=True))

    df["element_count"] = df["formula"].apply(
        lambda x: count_elements(str(x)) if pd.notnull(x) else 0
    )
    element_counts = df["element_count"].value_counts().sort_index()

    fig_height = 7
    fig_width = 8
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(fig_width / 2.54, fig_height / 2.54))
    plt.bar(element_counts.index, element_counts.values, color=color1, alpha=0.75)

    for i, (element_count, value) in enumerate(element_counts.items()):
        plt.text(
            element_count, value, str(value), ha="center", va="bottom", fontsize=7
        )

    plt.xlabel("Number of Elements", fontsize=10)
    plt.ylabel("Number of Formulas", fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_fig2c(tc_threshold=5, dataframe=None):
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "superconductors.csv")
    df = dataframe if dataframe is not None else pd.read_csv(DATA_PATH, low_memory=False)

    def preprocess_molecular_data(data, Tc_line=5):
        element_counts = {}
        for index, row in data.iterrows():
            if row["Tc_K"] >= Tc_line:
                from utils.scope import parse_scoped_formula
                for element in parse_scoped_formula(row["formula"], allow_zero=True, unrestricted=True):
                    element_counts[element] = element_counts.get(element, 0) + 1
        return element_counts

    def plot_element(ax, x, y, symbol, atomic_num, frequency, color_map, normalize):
        symb_fs = 9
        num_fs = 6
        Fre_fs = 6
        color = color_map(normalize(frequency)) if frequency else "lightgrey"
        ax.add_patch(
            plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor="black")
        )
        ax.text(
            x,
            y + 0.03,
            symbol,
            fontsize=symb_fs,
            ha="center",
            va="center",
            color="black",
        )
        ax.text(
            x - 0.18,
            y - 0.3,
            f"{atomic_num}",
            fontsize=num_fs,
            ha="center",
            va="center",
            color="black",
        )
        if frequency:
            ax.text(
                x,
                y + 0.37,
                f"{frequency}",
                fontsize=Fre_fs,
                ha="center",
                va="center",
                color="black",
            )

    def plot_elements(ax, element_counts, lanthanoid, actinoid, color_map, normalize):
        for atomic_num, symbol, group, period in ELEMENTS:
            if symbol in lanthanoid:
                x, y = 3 + lanthanoid.index(symbol), 9
            elif symbol in actinoid:
                x, y = 3 + actinoid.index(symbol), 10
            else:
                x, y = group, period
            plot_element(
                ax,
                x,
                y,
                symbol,
                atomic_num,
                element_counts.get(symbol, 0),
                color_map,
                normalize,
            )

    def plot_lanthanoid_actinoid_labels(ax):
        ax.text(2, 9, "Lanthanids", fontsize=8, ha="right", va="center", color="black")
        ax.text(2, 10, "Actinids", fontsize=8, ha="right", va="center", color="black")
        bg_color = "lightgrey"
        ax.text(
            3, 6 + 0.1, "La~\nLu", fontsize=8, ha="center", va="center", color="black"
        )
        ax.add_patch(
            plt.Rectangle(
                (3 - 0.5, 6 - 0.5), 1, 1, facecolor=bg_color, edgecolor="black"
            )
        )
        ax.text(
            3, 7 + 0.1, "Ac~\nLr", fontsize=8, ha="center", va="center", color="black"
        )
        ax.add_patch(
            plt.Rectangle(
                (3 - 0.5, 7 - 0.5), 1, 1, facecolor=bg_color, edgecolor="black"
            )
        )

    element_counts = preprocess_molecular_data(df, Tc_line=tc_threshold)
    lanthanoid = [
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    ]
    actinoid = [
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
    ]

    max_freq = max(element_counts.values(), default=1)
    normalize = Normalize(vmin=0, vmax=max_freq)
    color_map = cm.YlGn

    fig_height = 11
    fig_width = 16
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(fig_width / 2.54, fig_height / 2.54))
    plot_lanthanoid_actinoid_labels(ax)
    plot_elements(ax, element_counts, lanthanoid, actinoid, color_map, normalize)

    ax.set_xlim(0, 19)
    ax.set_ylim(0, 11)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis("off")

    cbar = fig.colorbar(
        cm.ScalarMappable(norm=normalize, cmap=color_map),
        ax=ax,
        orientation="vertical",
        pad=-0.01,
        shrink=0.9,
    )
    cbar.set_label("Frequency", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()
