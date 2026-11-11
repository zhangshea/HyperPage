"""Saving of cache-blind layouts over the incumbent when every layout uses the same hot-record share (no selection)."""
# [Paper] Fig. 3 (panels 1M at 1%, full corpus at 5% and 25%); the share-0 values are the pure page LRU of Section VII-B
# (14.22% to 40.02% over seven budgets; FREQ 20.06% and 24.69% at 25%).
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SERIES = [('SET-G', '#111111', 'o', '-'), ('ARC-G', '#111111', 's', '--'), ('CLQ-G', '#555555', '^', ':'),
          ('P05-G', '#555555', 'D', '-.')]
INK, MUTED, GRID = '#1f1f1e', '#5f5e5a', '#d9d8d2'


def load(paths):
    rows, meta = {}, None
    for p in paths:
        d = json.load(open(p))
        meta = meta or d
        rows.update(d['rows'])
    return rows, meta


def same_share_savings(rows, meta, budget, layouts, incumbent):
    """saving(X, rho) = 1 - sum_t mean_k R_X / sum_t mean_k R_incumbent, both at share rho, for every share."""
    mean = lambda name, s: np.mean([rows[f'{name}|{k}|{budget}|{s}']['per_step'] for k in meta['draws']], axis=0)
    out = {}
    for s in meta['shares']:
        base = mean(incumbent, s)
        out[s] = {x: float(100 * (1 - mean(x, s).sum() / base.sum())) for x in layouts}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel', nargs='+', action='append', required=True, metavar='ARG',
                    help='TITLE BUDGET FILE [FILE ...]: test replays over the full share grid at BUDGET pages; repeat '
                         'per panel, e.g. --panel "1M, 1%%" 833 test_1m_833.json')
    ap.add_argument('--layouts', default='SET-G,ARC-G,CLQ-G,P05-G', help='layouts to print (the figure draws the four '
                                                                         'cache-blind objectives)')
    ap.add_argument('--incumbent', default='MARGO')
    ap.add_argument('--out', default=None, help='figure file (PDF); without it only the table is printed')
    ap.add_argument('--height', type=float, default=1.32, help='figure height in inches (width 3.45)')
    a = ap.parse_args()

    layouts = a.layouts.split(',')
    panels = []
    for title, budget, *files in a.panel:
        rows, meta = load(files)
        t = same_share_savings(rows, meta, int(budget), layouts, a.incumbent)
        panels.append((title, t))
        print(title)
        for s, v in t.items():
            print(f'  share {s:4}: ' + '  '.join(f'{x} {y:6.2f}' for x, y in v.items()))
    if not a.out:
        return
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK,
                         'ytick.color': INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                         'pdf.fonttype': 42, 'mathtext.fontset': 'stix'})
    fig, axs = plt.subplots(1, len(panels), figsize=(3.45, a.height), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (title, t) in zip(axs, panels):
        shares = sorted(t)
        ax.axhline(0, color=INK, lw=0.7, zorder=1)
        for name, col, mk, ls in SERIES:
            if name in layouts:
                ax.plot(shares, [t[s][name] for s in shares], color=col, marker=mk, ms=2.4, lw=0.85, ls=ls, mfc='white',
                        mec=col, mew=0.5, label=name, zorder=3)
        ax.set_title(title, color=INK, fontsize=7.5, pad=2)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['0', '.5', '1']); ax.set_xlim(-0.05, 1.05)
        ax.grid(axis='y', color=GRID, lw=0.4); ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
    axs[0].set_ylabel('saving (%)', labelpad=1)
    fig.supxlabel(r'hot-record share $\rho$', fontsize=7.5, color=INK, y=0.015)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.53, 1.05), handlelength=2.4,
               columnspacing=1.0, fontsize=7)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.77, bottom=0.235, wspace=0.1)
    fig.savefig(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
