"""Saving of HyperPage's deployment over the incumbent in consecutive blocks of test steps."""
# [Paper] Section VII-B, Transfer to another agent: on W4 a block of 41 test steps is one OLMo-3 checkpoint, and the
# saving moves by at most 1.21 points from the first checkpoint to the last.
import argparse
import json

import numpy as np

from hyperpage import selection
from _results import load_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', required=True, help='output of select_deployment.py')
    ap.add_argument('--test', nargs='+', required=True, help='test replay files')
    ap.add_argument('--block', type=int, default=41, help='steps per block (W4: test steps per checkpoint)')
    ap.add_argument('--incumbent', default='MARGO')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    sel = json.load(open(a.selection))
    rows, meta = load_rows(a.test)
    draws = meta['draws']
    out = {}
    for budget in meta['budgets']:
        m = sel['shares'][f'{a.incumbent}|{budget}']
        base = np.mean([rows[selection.row_key(a.incumbent, k, budget, m)]['per_step'] for k in draws], axis=0)
        picks = sel['deployment'][str(budget)]
        hp = np.mean([rows[selection.row_key(p['layout'], p['draw'], budget, p['share'])]['per_step'] for p in picks],
                     axis=0)
        # pooled saving per block: draw-mean reads summed over the block's steps
        per_block = [float(100 * (1 - hp[i:i + a.block].sum() / base[i:i + a.block].sum()))
                     for i in range(0, len(hp), a.block)]
        total = float(100 * (1 - hp.sum() / base.sum()))
        out[str(budget)] = dict(blocks=per_block, total=total, spread=max(per_block) - min(per_block),
                                first_to_last=per_block[-1] - per_block[0])
        print(f'budget {budget} pages: ' + ' '.join(f'{x:.2f}' for x in per_block) +
              f' (all steps {total:.2f}; last minus first {per_block[-1] - per_block[0]:+.2f})')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
