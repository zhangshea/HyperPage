"""Saving over the incumbent as the index grows: HyperPage and the cache-blind ARC-G and P05-G at 1% and 5%."""
# [Paper] Fig. 4 (1M, 5M, 10M and 21M records); the printed HyperPage values give the drops of the Scale paragraph of
# Section VII-B (6.1 points from 1M to 5M and 2.6 more to the full corpus at 1%).
import argparse
import json
from decimal import Decimal, ROUND_HALF_UP

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SERIES = [('HyperPage', '#111111', 'o', '-', 1.1), ('ARC-G', '#555555', 's', '--', 0.85),
          ('P05-G', '#555555', 'D', ':', 0.85)]
INK, MUTED, GRID = '#1f1f1e', '#5f5e5a', '#d9d8d2'


def fmt(x, places):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def saving(path, layout):
    """Saving of one layout in a saving_table.py output that holds one budget."""
    table = json.load(open(path))
    assert len(table) == 1, f'{path}: one budget per file'
    return next(iter(table.values()))[layout]['saving']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index', nargs=4, action='append', required=True, metavar=('LABEL', 'MILLIONS', 'AT_1', 'AT_5'),
                    help='index label, size in millions of records, and the saving_table.py outputs (--out) at 1%% and '
                         '5%% of that index; repeat per index, smallest first')
    ap.add_argument('--out', required=True, help='figure file (PDF)')
    ap.add_argument('--height', type=float, default=1.3, help='figure height in inches (width 3.45)')
    a = ap.parse_args()

    budgets = ('1%', '5%')
    x = [float(m) for _, m, _, _ in a.index]
    y = {b: {name: [saving(row[2 + i], name) for row in a.index] for name, *_ in SERIES} for i, b in enumerate(budgets)}
    for b in budgets:
        for name, v in y[b].items():
            print(f'{b} {name:10s} ' + ' '.join(fmt(s, 2) for s in v))
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK,
                         'ytick.color': INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                         'pdf.fonttype': 42, 'mathtext.fontset': 'stix'})
    fig, axs = plt.subplots(1, 2, figsize=(3.45, a.height), sharey=True)
    for ax, b in zip(axs, budgets):
        ax.axhline(0, color=INK, lw=0.7, zorder=1)
        for name, col, mk, ls, lw in SERIES:
            ax.plot(x, y[b][name], color=col, marker=mk, ms=2.6, lw=lw, ls=ls,
                    mfc=col if name == 'HyperPage' else 'white', mec=col, mew=0.5, label=name, zorder=3)
        for xi, yi in zip(x, y[b]['HyperPage']):
            ax.annotate(fmt(yi, 1), (xi, yi), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=6.3,
                        color=INK)
        ax.set_xscale('log'); ax.set_xticks(x); ax.set_xticklabels([row[0] for row in a.index]); ax.minorticks_off()
        ax.set_xlim(0.8, 26)
        ax.set_title(f'{b} of the index', color=INK, fontsize=7.5, pad=2)
        ax.set_ylim(0, 25); ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.grid(axis='y', color=GRID, lw=0.4); ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
    axs[0].set_ylabel('saving over MARGO (%)', labelpad=1)
    fig.supxlabel('index size (records)', fontsize=7.5, color=INK, y=0.015)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.53, 1.04), handlelength=2.4,
               columnspacing=1.2, fontsize=7)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.78, bottom=0.22, wspace=0.08)
    fig.savefig(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
