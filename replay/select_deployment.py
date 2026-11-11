"""Selection on validation replays: the hot-record share of every layout and HyperPage's deployed (layout, share)."""
# [Paper] Section VI-D (selection for a declared budget) and Section VII-A, Metrics (each layout gets its own share).
import argparse
import json

from hyperpage import selection
from _results import load_rows, layouts_in


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('validation', nargs='+', help='validation replay files (full share grid)')
    ap.add_argument('--incumbent', default='MARGO')
    ap.add_argument('--candidates', required=True,
                    help="HyperPage's candidate layouts besides the incumbent, comma separated")
    ap.add_argument('--warmup', type=int, default=10, help='validation steps that only warm the cache')
    ap.add_argument('--scored', type=int, default=20, help='validation steps scored after the warm-up')
    ap.add_argument('--out', required=True,
                    help='JSON with "shares" (layout|budget -> share), "deployment" and "test_shares"')
    a = ap.parse_args()

    rows, meta = load_rows(a.validation)
    draws, shares = meta['draws'], meta['shares']
    candidates = a.candidates.split(',')
    out = dict(shares={}, deployment={}, test_shares={})
    for budget in meta['budgets']:
        for layout in layouts_in(rows, budget):
            out['shares'][f'{layout}|{budget}'] = selection.validation_share(rows, layout, draws, budget, shares,
                                                                             a.warmup, a.scored)
        picks = [selection.deployment(rows, candidates, a.incumbent, k, budget, shares, a.warmup, a.scored)
                 for k in draws]
        out['deployment'][str(budget)] = [dict(draw=k, layout=p[0], share=p[1], validation_reads=p[2])
                                          for k, p in zip(draws, picks)]
        # shares to replay on test: the share of every layout, and the share of the deployed pair of every draw
        for layout in layouts_in(rows, budget):
            out['test_shares'][f'{layout}|{budget}'] = [out['shares'][f'{layout}|{budget}']]
        for layout, share, _ in picks:
            if share not in out['test_shares'][f'{layout}|{budget}']:
                out['test_shares'][f'{layout}|{budget}'].append(share)
        print(budget, [(p[0], p[1]) for p in picks])
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
