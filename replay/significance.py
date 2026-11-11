"""Paired moving-block bootstrap of every saving over the incumbent on the test steps."""
# [Paper] Section VII-A: paired bootstrap over the test steps with 2,000 resamples, also with moving blocks of 5, 10
# and 26 steps; a saving differs from zero at level 0.05 when it does so for every block length (Table III, Table VII).
import argparse
import json

import numpy as np

from hyperpage import selection
from _results import load_rows, layouts_in


def draw_mean(rows, layout, draws, budget, share):
    return np.mean([rows[selection.row_key(layout, k, budget, share)]['per_step'] for k in draws], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', required=True, help='output of select_deployment.py')
    ap.add_argument('--test', nargs='+', required=True, help='test replay files')
    ap.add_argument('--incumbent', default='MARGO')
    ap.add_argument('--blocks', default='1,5,10,26', help='block lengths in steps; 1 is the plain paired bootstrap')
    ap.add_argument('--resamples', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0x13528f0)
    ap.add_argument('--level', type=float, default=0.05)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    sel = json.load(open(a.selection))
    rows, meta = load_rows(a.test)
    draws = meta['draws']
    blocks = [int(x) for x in a.blocks.split(',')]
    out = {}
    for budget in meta['budgets']:
        share_of = lambda layout: sel['shares'][f'{layout}|{budget}']
        base = draw_mean(rows, a.incumbent, draws, budget, share_of(a.incumbent))
        series = {}
        for layout in layouts_in(rows, budget):
            if layout != a.incumbent and selection.row_key(layout, draws[0], budget, share_of(layout)) in rows:
                series[layout] = draw_mean(rows, layout, draws, budget, share_of(layout))
        # HyperPage: the reads per step of each draw's deployed (layout, share), averaged over the draws
        picks = sel['deployment'][str(budget)]
        series['HyperPage'] = np.mean([rows[selection.row_key(p['layout'], p['draw'], budget, p['share'])]['per_step']
                                       for p in picks], axis=0)
        column = {}
        print(f'budget {budget} pages (level {a.level}, blocks {a.blocks}):')
        for layout, steps in series.items():
            res = {}
            for block in blocks:
                low, high, differs = selection.block_bootstrap(steps, base, block, a.resamples, a.seed, a.level)
                res[block] = dict(percentile_low=low, percentile_high=high, differs_from_zero=differs)
            saving = 100.0 * (1.0 - steps.sum() / base.sum())
            every = all(r['differs_from_zero'] for r in res.values())
            column[layout] = dict(saving=float(saving), blocks=res, differs_at_every_block=every)
            failing = [b for b, r in res.items() if not r['differs_from_zero']]
            print(f'  {layout:28s} saving {saving:7.2f} %  differs from zero: '
                  f'{"yes" if every else "no (blocks " + ",".join(map(str, failing)) + ")"}')
        out[str(budget)] = column
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
