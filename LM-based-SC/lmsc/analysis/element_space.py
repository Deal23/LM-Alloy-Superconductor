import argparse
import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.patches import PathPatch, Polygon, Rectangle
from matplotlib.path import Path
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

CMAP = 'viridis'
FAMILIES = ['Alloy', 'Fe-based', 'Metal-nonmetal']
from ..composition import NONMETALS as NONMETAL
from ..paths import ROOT, DATASET, DATA_RESULTS

SOURCE = DATASET
FIG_DIR = DATA_RESULTS


def classify_family(elems):
    if 'Fe' in elems:
        return 'Fe-based'
    if not (elems & NONMETAL):
        return 'Alloy'
    return 'Metal-nonmetal'


def parse_elements(formula):
    elems = set()
    for sym, num in re.findall(r'([A-Z][a-z]*)([0-9.]*)', str(formula)):
        if num and float(num) == 0:
            continue
        elems.add(sym)
    return frozenset(elems)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    tc_col = 'Tc_K' if 'Tc_K' in df.columns else 'Tc'
    df['Tc'] = pd.to_numeric(df[tc_col], errors='coerce')
    df = df.dropna(subset=['Tc'])
    df['elems'] = df['formula'].apply(parse_elements)
    df['n_elem'] = df['elems'].apply(len)
    df['family'] = df['elems'].apply(classify_family)
    return df


def build_stats(df):
    elem_tc = {}
    pair_tc = {}
    for elems, tc in zip(df['elems'], df['Tc']):
        for e in elems:
            elem_tc.setdefault(e, []).append(tc)
        for a, b in itertools.combinations(sorted(elems), 2):
            pair_tc.setdefault((a, b), []).append(tc)
    return elem_tc, pair_tc


def cluster_order(mat):
    X = np.nan_to_num(mat, nan=0.0)
    if len(X) < 3:
        return np.arange(len(X))
    try:
        Z = linkage(pdist(X), method='average', optimal_ordering=True)
        return leaves_list(Z)
    except Exception:
        return np.arange(len(X))


def draw_chord(ax, sub, norm, cmap, n_elems=25, top_chords=70):
    sub = sub[sub['n_elem'] >= 2]
    elem_tc, pair_tc = build_stats(sub)
    freq = {e: len(v) for e, v in elem_tc.items()}
    elems = sorted(freq, key=freq.get, reverse=True)[:n_elems]
    idx = {e: i for i, e in enumerate(elems)}
    n = len(elems)
    count = np.zeros((n, n))
    tcmax = np.full((n, n), np.nan)
    for (a, b), vals in pair_tc.items():
        if a in idx and b in idx:
            ia, ib = idx[a], idx[b]
            count[ia, ib] = count[ib, ia] = len(vals)
            tcmax[ia, ib] = tcmax[ib, ia] = np.max(vals)

    pairs = [((i, j), count[i, j], tcmax[i, j])
             for i in range(n) for j in range(i + 1, n) if count[i, j] > 0]
    pairs.sort(key=lambda x: x[1], reverse=True)
    shown = pairs[:top_chords]

    total = count.sum(axis=1)
    gap = np.deg2rad(2.2)
    available = 2 * np.pi - n * gap
    arcs = []
    a = np.pi / 2
    for i in range(n):
        span = total[i] / total.sum() * available if total.sum() else 0
        arcs.append([a, a + span])
        a += span + gap
    offsets = [arc[0] for arc in arcs]

    elem_max = {e: np.max(elem_tc[e]) for e in elems}

    for i in range(n):
        theta = np.linspace(arcs[i][0], arcs[i][1], 50)
        ax.plot(np.cos(theta), np.sin(theta), lw=6, solid_capstyle='butt',
                color=cmap(norm(elem_max[elems[i]])))
        mid = 0.5 * sum(arcs[i])
        rot = np.rad2deg(mid)
        ha = 'left'
        if 90 < rot % 360 < 270:
            rot += 180
            ha = 'right'
        r_lab = 1.06 if i % 2 == 0 else 1.15
        ax.text(r_lab * np.cos(mid), r_lab * np.sin(mid), elems[i],
                rotation=rot, rotation_mode='anchor', ha=ha, va='center',
                fontsize=10.5)

    for (i, j), c, t in shown:
        span_i = arcs[i][1] - arcs[i][0]
        span_j = arcs[j][1] - arcs[j][0]
        w_i = c / total[i] * span_i if total[i] else 0
        w_j = c / total[j] * span_j if total[j] else 0
        t1a, t1b = offsets[i], offsets[i] + w_i
        t2a, t2b = offsets[j], offsets[j] + w_j
        offsets[i] += w_i
        offsets[j] += w_j
        p1a = np.array([np.cos(t1a), np.sin(t1a)])
        p1b = np.array([np.cos(t1b), np.sin(t1b)])
        p2a = np.array([np.cos(t2a), np.sin(t2a)])
        p2b = np.array([np.cos(t2b), np.sin(t2b)])
        center = np.array([0.0, 0.0])
        verts = [p1a, center, p2a, p2b, center, p1b, p1a]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
                 Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=cmap(norm(t)),
                               edgecolor='none', alpha=0.6, zorder=1))

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)


def fig1_chord_grid(df, outdir):
    norm = Normalize(vmin=0, vmax=df['Tc'].max())
    cmap = plt.get_cmap(CMAP)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    fig.subplots_adjust(wspace=0.04)
    chords_map = {'Alloy': 100, 'Fe-based': 100, 'Metal-nonmetal': 100}
    for c, fam in enumerate(FAMILIES):
        ax = axes[c]
        sub = df[df['family'] == fam]
        draw_chord(ax, sub, norm, cmap, 25, chords_map[fam])
        ax.set_title(f'{fam} (n={len(sub)}, max {sub["Tc"].max():.0f} K)',
                     fontsize=15)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015,
                        pad=0.01)
    cbar.set_label('Pair max Tc (K)', fontsize=13)
    fig.savefig(os.path.join(outdir, 'data_element_chord.png'))
    plt.close(fig)


def draw_cell_stripes(ax, x, y, stats, norm, cmap, lw=0.35, edge='black'):
    tmin, tmed, tmax = stats
    for m, v in enumerate([tmax, tmed, tmin]):
        ax.add_patch(Rectangle((x, y + m / 3), 1, 1 / 3,
                               facecolor=cmap(norm(v)), edgecolor=edge,
                               lw=lw))


def fig2_family_violin(df, outdir):
    ks = [2, 3, 4, 5, 6, 7, 8]
    klabels = ['2', '3', '4', '5', '6', '7', '8+']
    colors = plt.get_cmap('tab10').colors
    nfam = len(FAMILIES)
    span = 0.8
    width = span / nfam
    fig, ax = plt.subplots(figsize=(14, 8))
    handles = []
    for fi, fam in enumerate(FAMILIES):
        sub = df[(df['family'] == fam) & (df['n_elem'] >= 2)]
        off = (fi - (nfam - 1) / 2) * width
        data, pos = [], []
        for j, k in enumerate(ks):
            vals = (sub.loc[sub['n_elem'] == k, 'Tc'].values if k < 8
                    else sub.loc[sub['n_elem'] >= 8, 'Tc'].values)
            if len(vals) >= 3:
                data.append(vals)
                pos.append(j + off)
        if not data:
            continue
        parts = ax.violinplot(data, positions=pos, widths=width * 0.95,
                              showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[fi])
            pc.set_alpha(0.8)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.4)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.0)
        order = np.argsort(pos)
        ax.plot(np.array(pos)[order], np.array([np.median(v) for v in data])[order],
                color=colors[fi], lw=1.8, marker='o', ms=3.5, zorder=5)
        handles.append(plt.Line2D([0], [0], color=colors[fi], lw=7,
                                  label=f'{fam} (n={len(sub)})'))
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(klabels, fontsize=12)
    ax.set_xlim(-0.6, len(ks) - 0.4)
    ax.set_ylim(0, df['Tc'].max() * 1.15)
    ax.set_xlabel('Number of elements', fontsize=13)
    ax.set_ylabel('Tc (K)', fontsize=13)
    ax.grid(alpha=0.25, lw=0.4, axis='y')
    ax.tick_params(length=0)
    ax.legend(handles=handles, fontsize=11, ncol=3, loc='upper center',
              frameon=False)
    fig.savefig(os.path.join(outdir, 'data_element_violin.png'))
    plt.close(fig)


def fig3_anchor_explore(df, outdir, anchors=('Ga', 'Bi', 'Sn', 'In'),
                        n_elems=25, min_triple=1):
    panels = []
    vmax, nmax = 0.0, 1
    for anchor in anchors:
        sub = df[df['elems'].apply(lambda s: anchor in s and len(s) >= 2
                                   and not (s & {'O', 'H'}))]
        pair_tc, triple_tc, co = {}, {}, {}
        for elems, tc in zip(sub['elems'], sub['Tc']):
            others = sorted(elems - {anchor})
            for e in others:
                co[e] = co.get(e, 0) + 1
                pair_tc.setdefault(e, []).append(tc)
            for a, b in itertools.combinations(others, 2):
                triple_tc.setdefault((a, b), []).append(tc)
        by_cnt = sorted(co, key=co.get, reverse=True)[:n_elems]
        by_tc = sorted(co, key=lambda e: max(pair_tc[e]),
                       reverse=True)[:n_elems]
        cand = list(dict.fromkeys(by_cnt + by_tc))
        cand.sort(key=lambda e: max(pair_tc[e]), reverse=True)
        elems = cand[:n_elems]
        idx = {e: i for i, e in enumerate(elems)}
        n = len(elems)
        mat_min = np.full((n, n), np.nan)
        mat_med = np.full((n, n), np.nan)
        mat_max = np.full((n, n), np.nan)
        mat_cnt = np.zeros((n, n))
        diag = {}
        for e, vals in pair_tc.items():
            if e in idx:
                diag[idx[e]] = (np.min(vals), np.median(vals), np.max(vals))
        for (a, b), vals in triple_tc.items():
            if a in idx and b in idx:
                ia, ib = idx[a], idx[b]
                mat_cnt[ia, ib] = mat_cnt[ib, ia] = len(vals)
                if len(vals) >= min_triple:
                    for m, v in ((mat_min, np.min(vals)),
                                 (mat_med, np.median(vals)),
                                 (mat_max, np.max(vals))):
                        m[ia, ib] = m[ib, ia] = v
        diag_max = np.full(n, -1.0)
        for e, vals in pair_tc.items():
            if e in idx:
                diag_max[idx[e]] = np.max(vals)
        order = np.argsort(-diag_max)
        mats = [m[order][:, order] for m in (mat_min, mat_med, mat_max,
                                             mat_cnt)]
        elems = [elems[i] for i in order]
        remap = {i: int(np.where(order == i)[0][0]) for i in range(n)}
        diag = {remap[i]: v for i, v in diag.items()}
        diag_cnt = {remap[i]: co[el] for el, i in idx.items()}
        dmax = max((v[2] for v in diag.values()), default=0.0)
        vmax = max(vmax, np.nanmax(mats[2]), dmax, float(sub['Tc'].max()))
        nmax = max(nmax, int(mats[3].max()))
        tvals = sub['Tc'].values
        stats_all = (np.min(tvals), np.median(tvals), np.max(tvals))
        panels.append((anchor, len(sub), elems, mats, diag, diag_cnt,
                       stats_all))

    cmap = plt.get_cmap(CMAP)
    grey = LinearSegmentedColormap.from_list(
        'grey_trunc', plt.get_cmap('Greys')(np.linspace(0.15, 1.0, 256)))
    vmax = np.ceil(vmax / 5) * 5
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    gnorm = Normalize(vmin=0, vmax=np.log10(nmax))
    fig, axes = plt.subplots(2, 2, figsize=(17, 15))
    fig.subplots_adjust(wspace=0.05, hspace=0.12, right=0.85)
    for ax, (anchor, nsub, elems, mats, diag, diag_cnt,
             stats_all) in zip(axes.ravel(), panels):
        mat_min, mat_med, mat_max, mat_cnt = mats
        n = len(elems)
        ax.set_facecolor('#f4f4f4')
        for i in range(n):
            for j in range(n):
                if j > i and not np.isnan(mat_med[i, j]):
                    draw_cell_stripes(ax, j - 0.5, i - 0.5,
                                      (mat_min[i, j], mat_med[i, j],
                                       mat_max[i, j]), norm, cmap, lw=0.4)
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor='black',
                                           lw=1.6))
                elif j < i and mat_cnt[i, j] > 0:
                    g = grey(gnorm(np.log10(mat_cnt[i, j])))
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=g, edgecolor='black',
                                           lw=0.4))
        for i, stats in diag.items():
            x0 = i - 0.5
            y0 = i - 0.5
            tri_up = Polygon([(x0, y0), (x0 + 1, y0), (x0 + 1, y0 + 1)],
                             closed=True, transform=ax.transData)
            for m, v in enumerate([stats[2], stats[1], stats[0]]):
                r = Rectangle((x0, y0 + m / 3), 1, 1 / 3,
                              facecolor=cmap(norm(v)), edgecolor='none')
                ax.add_patch(r)
                r.set_clip_path(tri_up)
            c = diag_cnt.get(i, 0)
            g = grey(gnorm(np.log10(c))) if c > 0 else '#f4f4f4'
            ax.add_patch(Polygon([(x0, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)],
                                 closed=True, facecolor=g, edgecolor='none'))
            ax.add_patch(Polygon([(x0, y0), (x0 + 1, y0), (x0 + 1, y0 + 1)],
                                 closed=True, fill=False, edgecolor='black',
                                 lw=0.5))
            ax.add_patch(Polygon([(x0, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)],
                                 closed=True, fill=False, edgecolor='black',
                                 lw=0.5))
            ax.add_patch(Rectangle((x0, y0), 1, 1, fill=False,
                                   edgecolor='black', lw=1.6))
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(elems, rotation=90, fontsize=10)
        ax.set_yticklabels(elems, fontsize=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        mn, md, mx = stats_all
        ax.set_title(f'{anchor}  n={nsub}   Tc min/med/max '
                     f'{mn:.0f}/{md:.0f}/{mx:.0f} K', fontsize=14)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax1 = fig.add_axes([0.87, 0.34, 0.012, 0.34])
    cbar = fig.colorbar(sm, cax=cax1)
    cbar.ax.yaxis.set_label_position('left')
    cbar.set_ticks([t for t in [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
                    if t <= vmax])
    cbar.set_label('Tc (K)', fontsize=11)
    sm2 = plt.cm.ScalarMappable(cmap=grey, norm=gnorm)
    cax2 = fig.add_axes([0.92, 0.34, 0.012, 0.34])
    cbar2 = fig.colorbar(sm2, cax=cax2)
    ticks = [t for t in [1, 3, 10, 30, 100, 300, 1000] if t <= nmax]
    cbar2.set_ticks(np.log10(ticks))
    cbar2.set_ticklabels([str(t) for t in ticks])
    cbar2.set_label('n compounds', fontsize=11)
    fig.savefig(os.path.join(outdir, 'data_element_anchor.png'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=str(SOURCE))
    parser.add_argument('--outdir', default=str(FIG_DIR))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.csv)
    print(f'Loaded {len(df)} records')
    print(df['family'].value_counts().to_string())

    fig1_chord_grid(df, args.outdir)
    print('1/3 data_element_chord.png')
    fig2_family_violin(df, args.outdir)
    print('2/3 data_element_violin.png')
    fig3_anchor_explore(df, args.outdir)
    print('3/3 data_element_anchor.png')


if __name__ == '__main__':
    main()
