"""Rank correlations over the layouts of one column: residual training connectivity against validation reads, and
validation reads against test reads."""
# [Paper] Section VII-B: residual training connectivity ranks the validation reads of each W3 column's 18 layouts with
# Spearman's r_s of 0.06 to 0.92 (mean over draws), and validation replay ranks test reads at 0.92 to 1.00.
import argparse
import json

import numpy as np

from hyperpage import selection
from _results import load_rows


def average_ranks(x):
    """Ranks 1..n with ties sharing their mean rank."""
    x = np.asarray(x, float); o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def spearman(x, y):
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--connectivity', required=True, help='output of residual_connectivity.py for the column')
    ap.add_argument('--validation', nargs='+', required=True, help='validation replays of the column (full share grid)')
    ap.add_argument('--test', nargs='+', required=True, help='test replays of every layout at its validation share')
    ap.add_argument('--budget', type=int, required=True, help='budget of the column in pages')
    ap.add_argument('--share', type=float, required=True,
                    help='the column\'s common share (the incumbent\'s validation share, which fixes h)')
    ap.add_argument('--layouts', default=None, help='layouts to rank, comma separated (default: all in --connectivity)')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--scored', type=int, default=20)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    conn = json.load(open(a.connectivity))
    h = f'h{conn["hot_pages"]}'
    val, vmeta = load_rows(a.validation)
    test, _ = load_rows(a.test)
    draws, shares = vmeta['draws'], vmeta['shares']
    names = a.layouts.split(',') if a.layouts else sorted({key.split('|')[0] for key in conn['rows']}, key=str)
    arms = {}
    for name in names:
        # connectivity and validation reads at the common share, per draw
        o = [conn['rows'][f'{name}|{k}'][h] for k in draws]
        w = [selection.scored_reads(val, name, k, a.budget, a.share, a.warmup, a.scored) for k in draws]
        # validation reads and test reads at the layout's own validation share, draw means
        own = selection.validation_share(val, name, draws, a.budget, shares, a.warmup, a.scored)
        w_own = float(np.mean([selection.scored_reads(val, name, k, a.budget, own, a.warmup, a.scored) for k in draws]))
        t_own = float(np.mean([selection.total_reads(test, name, k, a.budget, own) for k in draws]))
        arms[name] = dict(connectivity=o, validation_at_share=w, own_share=own, validation_at_own=w_own,
                          test_at_own=t_own)
    per_draw = [spearman([arms[x]['connectivity'][i] for x in names],
                         [arms[x]['validation_at_share'][i] for x in names])
                for i in range(len(draws))]
    val_test = spearman([arms[x]['validation_at_own'] for x in names], [arms[x]['test_at_own'] for x in names])
    out = dict(layouts=len(names), connectivity_vs_validation=per_draw,
               connectivity_vs_validation_mean=float(np.mean(per_draw)), validation_vs_test=val_test, arms=arms)
    print(f'{len(names)} layouts: r_s(connectivity, validation reads) per draw '
          f'{", ".join(f"{x:.2f}" for x in per_draw)}, mean {np.mean(per_draw):.2f}; '
          f'r_s(validation reads, test reads) {val_test:.2f}')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
