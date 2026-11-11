"""Selection of the result-cache share on validation and the savings with a result cache in front, on test."""
# [Paper] Section VII-C: an exact-string result cache of at most 1/64 of the budget saves MARGO 16.82% to 25.17% of its
# reads and leaves HyperPage 12.14% to 20.59% ahead (here: the 1M index).
# select: the incumbent takes the (share, phi) with the fewest draw-averaged validation reads; HyperPage keeps its
#         deployed layout of every draw and takes per draw the (share, phi) with the fewest validation reads; ties go to
#         the smaller phi, then to the smaller share.
# read:   per draw, the incumbent with the cache against the incumbent without it, and HyperPage with the cache against
#         the incumbent with it.
import argparse
import json

import numpy as np

from hyperpage import selection
from _results import load_rows


def load(paths):
    """Rows of result-cache replays keyed (row key, phi)."""
    rows, meta = {}, None
    for path in paths:
        d = json.load(open(path))
        meta = meta or d
        for key, row in d['rows'].items():
            rows[(key, d['rc_share'])] = row
    return rows, meta


def scored(rows, layout, draw, budget, share, phi, warmup, n):
    return float(np.sum(rows[(selection.row_key(layout, draw, budget, share), phi)]['per_step'][warmup:warmup + n]))


def select(a):
    rows, meta = load(a.validation)
    draws, shares = meta['draws'], meta['shares']
    phis = sorted({key[1] for key in rows})
    sel = json.load(open(a.selection))
    out = {}
    for budget in meta['budgets']:
        mean_scored = lambda s, ph: float(np.mean([scored(rows, a.incumbent, k, budget, s, ph, a.warmup, a.scored)
                                                   for k in draws]))
        grid = [(mean_scored(s, ph), ph, s) for ph in phis for s in shares
                if (selection.row_key(a.incumbent, draws[0], budget, s), ph) in rows]
        v, ph, s = min(grid)
        out[f'incumbent|{budget}'] = [dict(draw=k, layout=a.incumbent, share=s, phi=ph, validation_reads=v)
                                      for k in draws]
        out[f'HyperPage|{budget}'] = []
        for p in sel['deployment'][str(budget)]:
            k, layout = p['draw'], p['layout']
            v, ph, s = min((scored(rows, layout, k, budget, s, ph, a.warmup, a.scored), ph, s)
                           for ph in phis for s in shares)
            out[f'HyperPage|{budget}'].append(dict(draw=k, layout=layout, share=s, phi=ph, validation_reads=v))
        for arm in ('incumbent', 'HyperPage'):
            print(budget, arm, [(x['layout'], x['share'], x['phi']) for x in out[f'{arm}|{budget}']])
        if any(x['phi'] == max(phis) for arm in ('incumbent', 'HyperPage') for x in out[f'{arm}|{budget}']):
            print(f'budget {budget}: the largest phi of the grid was chosen; extend the grid')
    json.dump(out, open(a.out, 'w'), indent=1)


def read(a):
    rc = json.load(open(a.rc_selection))
    rows, meta = load(a.test)
    plain, _ = load_rows(a.reference_test)
    sel = json.load(open(a.selection))
    out = {}
    for budget in meta['budgets']:
        def total(x):
            key = (selection.row_key(x['layout'], x['draw'], budget, x['share']), x['phi'])
            return float(np.sum(rows[key]['per_step']))
        m0 = [selection.total_reads(plain, a.incumbent, p['draw'], budget, sel['shares'][f'{a.incumbent}|{budget}'])
              for p in rc[f'incumbent|{budget}']]
        m_rc = [total(x) for x in rc[f'incumbent|{budget}']]
        hp_rc = [total(x) for x in rc[f'HyperPage|{budget}']]
        cell = dict(incumbent_with_cache_vs_without=selection.saving_per_draw(m_rc, m0),
                    hyperpage_with_cache_vs_incumbent_with_cache=selection.saving_per_draw(hp_rc, m_rc))
        out[str(budget)] = cell
        for name, v in cell.items():
            print(f'budget {budget}: {name} {np.mean(v):.2f} % (draws {", ".join(f"{x:.2f}" for x in v)})')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('select', help='choose (share, phi) on validation')
    s.add_argument('--validation', nargs='+', required=True,
                   help='validation replays of replay_result_cache.py: the incumbent and the deployed layouts, every '
                        'share and every phi of the grid (one file per phi)')
    s.add_argument('--selection', required=True, help='select_deployment.py output (deployed layout per draw)')
    s.add_argument('--out', required=True)
    r = sub.add_parser('read', help='savings on test')
    r.add_argument('--rc-selection', required=True, help='output of the select mode')
    r.add_argument('--test', nargs='+', required=True, help='test replays of the selected (layout, share, phi) cells')
    r.add_argument('--reference-test', nargs='+', required=True,
                   help='test replays without result cache that hold the incumbent at its validation share')
    r.add_argument('--selection', required=True, help='select_deployment.py output (the incumbent\'s share)')
    r.add_argument('--out', default=None)
    for p in (s, r):
        p.add_argument('--incumbent', default='MARGO')
        p.add_argument('--warmup', type=int, default=10)
        p.add_argument('--scored', type=int, default=20)
    a = ap.parse_args()
    select(a) if a.mode == 'select' else read(a)


if __name__ == '__main__':
    main()
