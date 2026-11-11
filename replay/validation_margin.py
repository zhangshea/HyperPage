"""Validation reads of layouts at their validation share against the incumbent at its own, as a saving in percent."""
# [Paper] Section VII-B: a cache-aware layout refined for one large budget of the full corpus, replayed at the other,
# reads 0.8% to 3.0% more than MARGO on validation steps.
import argparse
import json

import numpy as np

from hyperpage import selection
from _results import load_rows


def best_scored(rows, layout, draws, budget, shares, warmup, scored):
    """Fewest draw-averaged scored validation reads over the shares, and the share that gives them."""
    share = selection.validation_share(rows, layout, draws, budget, shares, warmup, scored)
    reads = np.mean([selection.scored_reads(rows, layout, k, budget, share, warmup, scored) for k in draws])
    return float(reads), share


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('validation', nargs='+', help='validation replays (full share grid) at one budget')
    ap.add_argument('--incumbent', default='MARGO')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--scored', type=int, default=20)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    rows, meta = load_rows(a.validation)
    draws, shares = meta['draws'], meta['shares']
    out = {}
    for budget in meta['budgets']:
        base, base_share = best_scored(rows, a.incumbent, draws, budget, shares, a.warmup, a.scored)
        for layout in sorted({key.split('|')[0] for key in rows if key.split('|')[2] == str(budget)}):
            if layout == a.incumbent:
                continue
            reads, share = best_scored(rows, layout, draws, budget, shares, a.warmup, a.scored)
            out[f'{layout}|{budget}'] = dict(share=share, saving=100 * (1 - reads / base), incumbent_share=base_share)
            print(f'budget {budget} pages: {layout} at share {share} saves {100 * (1 - reads / base):.2f}% of the '
                  f'validation reads of {a.incumbent} at share {base_share}')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
