"""Budget at which the incumbent, with its validation share at every budget, reads as few test pages as HyperPage."""
# [Paper] Table III, last row: per draw, the first grid budget g >= D at which MARGO (its share chosen on validation at
# g, per draw) reads no more test pages than HyperPage's deployment at D, and the grid point before it; the bracket is
# [min over draws of previous / D, max over draws of g / D].
import argparse
import json

from hyperpage import selection
from _results import load_rows

# budget grid as fractions of the index (1M index and W4)
GRID = ('0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.03,0.035,0.04,0.045,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.14,'
        '0.16,0.18,0.20,0.25,0.26,0.27,0.28,0.29,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90,1.0')
GRID_FULL_CORPUS = ('0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.03,0.035,0.04,0.045,0.05,0.0525,0.055,0.0575,0.06,'
                    '0.07,0.08,0.09,0.10,0.12,0.14,0.149687,0.155,0.16,0.165,0.17,0.18,0.19,0.20,0.22,0.24,0.25,0.26,'
                    '0.27,0.28,0.30,0.35,0.40,0.45,0.50,0.60')


def grid_pages(grid, index_pages):
    """Budget in pages of every grid fraction f: round(f * index pages)."""
    return {f: int(round(index_pages * float(f))) for f in grid.split(',')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--grid', default='1m',
                    help='budget grid as comma-separated fractions of the index, or "1m" (default, the grid of the 1M '
                         'index and W4: ' + GRID + ') or "full-corpus" (' + GRID_FULL_CORPUS + ')')
    ap.add_argument('--index-pages', type=int, required=True,
                    help='pages that the grid fractions refer to (budgets are round(f * pages); 83334 at 1M, 1751278 '
                         'for the full-corpus budgets of the paper)')
    ap.add_argument('--list-budgets', action='store_true', help='print the grid in pages, comma separated, and exit')
    ap.add_argument('--declared', type=float, help='declared budget D as a fraction of the index (a grid point)')
    ap.add_argument('--validation', nargs='+', help='validation replays of the incumbent over the grid (all shares)')
    ap.add_argument('--test', nargs='+', help='test replays of the incumbent over the grid (all shares)')
    ap.add_argument('--selection', help='select_deployment.py output at the declared budget')
    ap.add_argument('--deployed-test', nargs='+', 
                    help='test replays that hold the deployed (layout, share) of every draw')
    ap.add_argument('--incumbent', default='MARGO')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--scored', type=int, default=20)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    pages = grid_pages({'1m': GRID, 'full-corpus': GRID_FULL_CORPUS}.get(a.grid, a.grid), a.index_pages)
    if a.list_budgets:
        print(','.join(str(p) for p in pages.values()))
        return
    val, vmeta = load_rows(a.validation)
    test, _ = load_rows(a.test)
    draws, shares = vmeta['draws'], vmeta['shares']
    sel = json.load(open(a.selection))
    declared_pages = pages[min(pages, key=lambda f: abs(float(f) - a.declared))]
    picks = sel['deployment'][str(declared_pages)]
    deployed, _ = load_rows(a.deployed_test)
    out = dict(declared=a.declared, declared_pages=declared_pages, draws={})
    lows, highs = [], []
    for p in picks:
        k = p['draw']
        target = selection.total_reads(deployed, p['layout'], k, declared_pages, p['share'])
        # the incumbent's test reads at every grid budget, at its share chosen on validation for this draw
        reads, chosen = {}, {}
        for f, b in pages.items():
            s = selection.validation_share_of_draw(val, a.incumbent, k, b, shares, a.warmup, a.scored)
            chosen[f] = s
            reads[float(f)] = selection.total_reads(test, a.incumbent, k, b, s)
        reached = [g for g in sorted(reads) if g >= a.declared - 1e-9 and reads[g] <= target]
        if not reached:
            out['draws'][k] = dict(deployed=[p['layout'], p['share']], target=target, bracket=None)
            print(f'draw {k}: the incumbent does not reach {target:.0f} reads on the grid')
            highs.append(None)
            continue
        if reached[0] == min(reads):
            low, high = 0.0, reached[0] / a.declared
        else:
            low, high = selection.budget_bracket(reads, target, a.declared)
        lows.append(low); highs.append(high)
        out['draws'][k] = dict(deployed=[p['layout'], p['share']], target=target, bracket=[low, high],
                               incumbent_shares=chosen)
        print(f'draw {k}: {p["layout"]} at share {p["share"]} reads {target:.0f}; the incumbent matches between '
              f'{low:.4g} and {high:.4g} times the declared budget')
    if all(h is not None for h in highs):
        low, high = min(lows), max(highs)
        out['bracket'] = [low, high]
        print(f'union over the draws: {low:.2f} to {high:.2f} times the declared budget')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
