"""Saving in page reads over the incumbent on test replays: every layout at its validation share, and HyperPage."""
# [Paper] Table III (rows of the compared layouts and of HyperPage) and Table VII (HyperPage's candidates).
# [Paper] Section VII-C, Other workloads: with --incumbent-per-draw, MARGO's share is chosen on each draw (W1, W2).
import argparse
import json

import numpy as np

from hyperpage import selection
from _results import load_rows, layouts_in


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', required=True, help='output of select_deployment.py')
    ap.add_argument('--test', nargs='+', required=True, help='test replay files')
    ap.add_argument('--incumbent', default='MARGO')
    ap.add_argument('--out', default=None, help='optional JSON with the unrounded values')
    ap.add_argument('--incumbent-per-draw', nargs='+', default=None, metavar='VALIDATION',
                    help="validation replay files: choose the incumbent's share on each draw (fewest scored validation "
                         "reads, ties to the smaller share) instead of the share of --selection; the test files must "
                         "hold the incumbent at these shares")
    a = ap.parse_args()

    sel = json.load(open(a.selection))
    rows, meta = load_rows(a.test)
    draws = meta['draws']
    if a.incumbent_per_draw:
        val_rows, val_meta = load_rows(a.incumbent_per_draw)
    table = {}
    for budget in meta['budgets']:
        share_of = lambda layout: sel['shares'][f'{layout}|{budget}']
        if a.incumbent_per_draw:
            # the incumbent's own share on every draw; each saving of a draw is taken against it
            incumbent_share = [selection.validation_share_of_draw(val_rows, a.incumbent, k, budget, val_meta['shares'])
                               for k in draws]
            base = [selection.total_reads(rows, a.incumbent, k, budget, s) for k, s in zip(draws, incumbent_share)]
        else:
            incumbent_share = share_of(a.incumbent)
            base = [selection.total_reads(rows, a.incumbent, k, budget, share_of(a.incumbent)) for k in draws]
        column = {a.incumbent: dict(reads_per_step=float(np.sum(base)) / len(draws) / meta['n_steps'],
                                    share=incumbent_share)}
        for layout in layouts_in(rows, budget):
            if layout == a.incumbent:
                continue
            key = selection.row_key(layout, draws[0], budget, share_of(layout))
            if key not in rows:
                continue
            totals = [selection.total_reads(rows, layout, k, budget, share_of(layout)) for k in draws]
            per_draw = selection.saving_per_draw(totals, base)
            column[layout] = dict(saving=selection.pooled_saving(totals, base), share=share_of(layout),
                                  per_draw=per_draw, std=float(np.std(per_draw, ddof=1)) if len(draws) > 1 else 0.0)
        picks = sel['deployment'][str(budget)]
        totals = [selection.total_reads(rows, p['layout'], p['draw'], budget, p['share']) for p in picks]
        per_draw = selection.saving_per_draw(totals, base)
        column['HyperPage'] = dict(saving=selection.pooled_saving(totals, base),
                                   picks=[(p['layout'], p['share']) for p in picks], per_draw=per_draw,
                                   std=float(np.std(per_draw, ddof=1)) if len(draws) > 1 else 0.0)
        table[str(budget)] = column
        print(f'budget {budget} pages: {a.incumbent} reads {column[a.incumbent]["reads_per_step"]:.1f} per step '
              f'at share {incumbent_share}')
        for layout, cell in column.items():
            if layout != a.incumbent:
                print(f'  {layout:28s} saving {selection.round_half_up(cell["saving"], 1):6.1f} %')
    if a.out:
        json.dump(table, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
