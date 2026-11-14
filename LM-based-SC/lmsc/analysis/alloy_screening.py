import argparse
import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde

# Journal-style defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

CMAP_BWR = LinearSegmentedColormap.from_list('bwr', ['#2166ac', '#f7f7f7', '#b2182b'])
C_BLUE = '#2166ac'
C_RED = '#b2182b'
from ..paths import ROOT, PREDICTIONS

DATA_DIR = PREDICTIONS / 'alloy_screening'
FIG_DIR = PREDICTIONS


def parse_formula(formula):
    formula = str(formula)
    ratios = {}
    for element, ratio in re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', formula):
        ratios[element] = ratios.get(element, 0) + (float(ratio) if ratio else 1)
    total = sum(ratios.values())
    return {k: v / total for k, v in ratios.items()}


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['calculated'] == 'yes'].copy()
    df['Predicted_Tc'] = pd.to_numeric(df['Predicted_Tc'], errors='coerce')
    df = df.dropna(subset=['Predicted_Tc'])
    prefix = re.match(r"(\d+)", os.path.basename(csv_path)).group(1)
    df['Type'] = f'{prefix}-Alloy'
    return df


def prepare_compositions(df2, df3):
    rows = []
    for df in (df2, df3):
        for _, r in df.iterrows():
            comp = parse_formula(r['Elements'])
            if not comp:
                continue
            rows.append({
                'type': r['Type'],
                'Tc': r['Predicted_Tc'],
                'comp': comp,
                'elements_label': r['elements'],
            })
    return pd.DataFrame(rows)


def plot_kde(ax, data, color, label):
    if len(data) < 2:
        return
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 500)
    y = kde(x)
    ax.plot(x, y, color=color, lw=2, label=label)
    ax.fill_between(x, y, color=color, alpha=0.15)


def plot_tc_distribution(ax, df2, df3):
    """(a) Clean KDE comparison of predicted Tc for 2-Alloy vs 3-Alloy."""
    tc2 = df2['Predicted_Tc'].values
    tc3 = df3['Predicted_Tc'].values

    plot_kde(ax, tc2, C_BLUE, f'2-Alloy (n={len(tc2)})')
    plot_kde(ax, tc3, C_RED, f'3-Alloy (n={len(tc3)})')

    ax.set_xlabel('Predicted Tc (K)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Predicted Tc distribution')
    ax.legend(loc='upper right', frameon=False)


def cluster_order(mat):
    """Return a leaf ordering that groups similar rows/cols together."""
    X = np.nan_to_num(mat, nan=0.0)
    if len(X) < 3:
        return np.arange(len(X))
    try:
        Z = linkage(pdist(X), method='average', optimal_ordering=True)
        return leaves_list(Z)
    except Exception:
        return np.arange(len(X))


def compute_pair_stats(df):
    """
    Return pair max-Tc dict, element max-Tc dict, and the third element that
    achieves each pair max (ternary only; None for binary pairs).
    Pair values are derived from the nominal element combination labels:
    - binary: the pair in the label gets the row's Tc
    - ternary: all three pairs in the label get the row's Tc
    """
    pair_max = {}
    pair_third = {}
    elem_max = {}
    for _, r in df.iterrows():
        t = r['Tc']
        elems = r['elements_label'].split('-')
        for e in elems:
            elem_max[e] = max(elem_max.get(e, -np.inf), t)
        for a, b in itertools.combinations(elems, 2):
            key = tuple(sorted((a, b)))
            if t > pair_max.get(key, -np.inf):
                pair_max[key] = t
                if len(elems) == 3:
                    pair_third[key] = [e for e in elems if e not in (a, b)][0]
    return pair_max, elem_max, pair_third


def build_pair_matrix(elems, pair_max, pair_third=None):
    """
    Build a symmetric pair maximum-Tc matrix for the given element list.
    If pair_third is provided, also build the matching third-element matrix.
    """
    idx = {e: i for i, e in enumerate(elems)}
    n = len(elems)
    mat = np.full((n, n), np.nan)
    third = np.full((n, n), '', dtype=object) if pair_third is not None else None
    for (a, b), val in pair_max.items():
        if a in idx and b in idx:
            ia, ib = idx[a], idx[b]
            mat[ia, ib] = val
            mat[ib, ia] = val
            if third is not None:
                label = pair_third.get((a, b), '')
                third[ia, ib] = label
                third[ib, ia] = label
    if third is not None:
        return mat, third
    return mat


def plot_pair_heatmap(ax, mat, elems, title, norm, show_labels=True, fontsize=8,
                      third_mat=None):
    """Plot a symmetric pair heatmap with optional labels and cell annotations."""
    mat_plot = np.ma.masked_where(np.isnan(mat), mat)
    im = ax.imshow(mat_plot, cmap=CMAP_BWR, aspect='auto', norm=norm)

    n = len(elems)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    if show_labels:
        ax.set_xticklabels(elems, rotation=90, ha='center', fontsize=fontsize)
        ax.set_yticklabels(elems, fontsize=fontsize)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_title(title)

    # Annotate cells with the third element that achieves the pair max
    if third_mat is not None:
        mid = 0.5 * (norm.vmin + norm.vmax)
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]) and third_mat[i, j]:
                    color = 'white' if mat[i, j] > mid else 'black'
                    ax.text(j, i, third_mat[i, j], ha='center', va='center',
                            fontsize=5, color=color)
    return im



def plot_binary_vs_ternary_pairs(ax, df_binary, df_ternary):
    """
    (d) Macro comparison: binary pair max Tc vs ternary pair max Tc.
    Each point is an element pair present in both sets. Color encodes the
    relative gain (%) on a signed-log scale; dotted reference lines mark
    +5/10/20/50/100% improvement levels.
    """
    bin_max, _, _ = compute_pair_stats(df_binary)
    ter_max, _, _ = compute_pair_stats(df_ternary)

    common = set(bin_max) & set(ter_max)
    x = np.array([bin_max[k] for k in common])
    y = np.array([ter_max[k] for k in common])
    pct = (y - x) / x * 100.0

    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='y = x')

    # Percentage reference lines: y = (1 + p) * x
    for p in [0.05, 0.10, 0.20, 0.50, 1.00]:
        xs = np.linspace(0, lim / (1 + p), 50)
        ax.plot(xs, xs * (1 + p), color='gray', lw=0.7, ls=':', alpha=0.7,
                zorder=1)
        ax.text(lim / (1 + p), lim * 0.99, f'+{int(p * 100)}%',
                ha='right', va='top', fontsize=8, color='gray')

    # Signed-log transform so the wide percentage range stays readable
    def trans(p):
        return np.sign(p) * np.log1p(np.abs(p))

    tvals = trans(pct)
    norm = Normalize(vmin=trans(pct.min()), vmax=trans(pct.max()))

    sc = ax.scatter(x, y, c=tvals, cmap=CMAP_BWR, norm=norm,
                    s=30, edgecolors='k', linewidth=0.3, zorder=2)

    # Colorbar with human-readable percentage ticks
    tick_pcts = [p for p in [-5, 0, 10, 50, 100, 300, 1000, 3000]
                 if pct.min() <= p <= pct.max()]
    cbar = plt.colorbar(sc, ax=ax, fraction=0.06, pad=0.02)
    cbar.set_ticks([trans(p) for p in tick_pcts])
    cbar.set_ticklabels(['0%' if p == 0 else f'{p:+d}%' for p in tick_pcts])
    cbar.set_label('Tc gain from third element (%)')

    ax.set_xlabel('Binary pair max Tc (K)')
    ax.set_ylabel('Ternary pair max Tc (K)')
    ax.set_title('(d) Third-element Tc gain: ternary vs binary pairs')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(loc='upper left', frameon=False)


def main():
    parser = argparse.ArgumentParser(description='Visualize global alloy Tc composition relationships.')
    parser.add_argument('--csv2', default=str(DATA_DIR / '2alloy.csv'), help='Path to 2-Alloy CSV')
    parser.add_argument('--csv3', default=str(DATA_DIR / '3alloy.csv'), help='Path to 3-Alloy CSV')
    parser.add_argument('--outdir', default=str(FIG_DIR), help='Output directory')
    parser.add_argument('--outfile', default='screening_alloy_overview.png', help='Output figure filename')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df2 = load_data(args.csv2)
    df3 = load_data(args.csv3)
    df_all = prepare_compositions(df2, df3)

    df_binary = df_all[df_all['type'] == '2-Alloy']
    df_ternary = df_all[df_all['type'] == '3-Alloy']

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.96, top=0.95, bottom=0.07,
                          wspace=0.28, hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_tc_distribution(ax1, df2, df3)

    # Binary pair section
    pair_max_bin, elem_max_bin, _ = compute_pair_stats(df_binary)

    n_local = 25
    # Local subset: elements involved in the top pair max values, then cluster within them
    elem_pair_max_bin = {}
    for (a, b), val in pair_max_bin.items():
        elem_pair_max_bin[a] = max(elem_pair_max_bin.get(a, -np.inf), val)
        elem_pair_max_bin[b] = max(elem_pair_max_bin.get(b, -np.inf), val)
    local_elems_bin = sorted(elem_pair_max_bin.keys(), key=lambda e: elem_pair_max_bin[e], reverse=True)[:n_local]
    mat_local_bin = build_pair_matrix(local_elems_bin, pair_max_bin)
    order_local_bin = cluster_order(mat_local_bin)
    mat_local_bin = mat_local_bin[order_local_bin][:, order_local_bin]
    local_elems_bin = [local_elems_bin[i] for i in order_local_bin]

    # Global matrix: local elements first (same order as b), then the rest clustered
    all_elems_bin = sorted(elem_max_bin.keys(), key=lambda e: elem_max_bin[e], reverse=True)
    mat_rest_bin = build_pair_matrix(all_elems_bin, pair_max_bin)
    order_rest_bin = cluster_order(mat_rest_bin)
    rest_elems_bin = [all_elems_bin[i] for i in order_rest_bin if all_elems_bin[i] not in set(local_elems_bin)]
    final_order_bin = local_elems_bin + rest_elems_bin
    mat_global_bin = build_pair_matrix(final_order_bin, pair_max_bin)

    vmax_bin = np.nanmax(mat_global_bin)
    norm_bin = Normalize(vmin=0, vmax=vmax_bin)

    ax_b = fig.add_subplot(gs[0, 1])
    plot_pair_heatmap(ax_b, mat_local_bin, local_elems_bin,
                      '(b) Binary pair max Tc (local)', norm_bin,
                      show_labels=True, fontsize=8)

    ax_c = fig.add_subplot(gs[0, 2])
    im_c = plot_pair_heatmap(ax_c, mat_global_bin, final_order_bin,
                             '(c) Binary pair max Tc (global)', norm_bin,
                             show_labels=False)
    fig.colorbar(im_c, ax=ax_c, fraction=0.06, pad=0.02, label='Max Tc (K)')

    # Macro comparison d
    ax_d = fig.add_subplot(gs[1, 0])
    plot_binary_vs_ternary_pairs(ax_d, df_binary, df_ternary)

    # Ternary pair section
    pair_max_ter, elem_max_ter, pair_third_ter = compute_pair_stats(df_ternary)

    # Local subset: elements involved in the top pair max values, then cluster within them
    elem_pair_max_ter = {}
    for (a, b), val in pair_max_ter.items():
        elem_pair_max_ter[a] = max(elem_pair_max_ter.get(a, -np.inf), val)
        elem_pair_max_ter[b] = max(elem_pair_max_ter.get(b, -np.inf), val)
    local_elems_ter = sorted(elem_pair_max_ter.keys(), key=lambda e: elem_pair_max_ter[e], reverse=True)[:n_local]
    mat_local_ter, third_local_ter = build_pair_matrix(local_elems_ter, pair_max_ter,
                                                       pair_third=pair_third_ter)
    order_local_ter = cluster_order(mat_local_ter)
    mat_local_ter = mat_local_ter[order_local_ter][:, order_local_ter]
    third_local_ter = third_local_ter[order_local_ter][:, order_local_ter]
    local_elems_ter = [local_elems_ter[i] for i in order_local_ter]

    # Global matrix: local elements first (same order as e), then the rest clustered
    all_elems_ter = sorted(elem_max_ter.keys(), key=lambda e: elem_max_ter[e], reverse=True)
    mat_rest_ter = build_pair_matrix(all_elems_ter, pair_max_ter)
    order_rest_ter = cluster_order(mat_rest_ter)
    rest_elems_ter = [all_elems_ter[i] for i in order_rest_ter if all_elems_ter[i] not in set(local_elems_ter)]
    final_order_ter = local_elems_ter + rest_elems_ter
    mat_global_ter = build_pair_matrix(final_order_ter, pair_max_ter)

    # Shared e/f scale: range of the values actually displayed (non-NaN cells)
    vmin_ter = np.nanmin(mat_global_ter)
    vmax_ter = np.nanmax(mat_global_ter)
    norm_ter = Normalize(vmin=vmin_ter, vmax=vmax_ter)

    ax_e = fig.add_subplot(gs[1, 1])
    plot_pair_heatmap(ax_e, mat_local_ter, local_elems_ter,
                      '(e) Ternary pair max Tc (local, label = third element)',
                      norm_ter, show_labels=True, fontsize=8,
                      third_mat=third_local_ter)

    ax_f = fig.add_subplot(gs[1, 2])
    im_f = plot_pair_heatmap(ax_f, mat_global_ter, final_order_ter,
                             '(f) Ternary pair max Tc (global)', norm_ter,
                             show_labels=False)
    fig.colorbar(im_f, ax=ax_f, fraction=0.06, pad=0.02, label='Max Tc (K)')

    plt.savefig(os.path.join(args.outdir, args.outfile))
    plt.close()

    print(f'Figure saved to: {os.path.join(args.outdir, args.outfile)}')


if __name__ == '__main__':
    main()
