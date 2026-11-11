"""Exhaustive check of the instance of Proposition 3 (Tightness): every layout of the B x B grid instance is enumerated,
and the maximizers of the clique objective are compared with the minimizers of the connectivity."""
# [Paper] Proposition 3 (Tightness): on V = [B] x [B], the rows are sets of weight 1 + eps and every pair of cells in
# one column is a set of weight 1. The clique objective F_clq has a unique maximizer pi_c up to renaming pages, and
# 2 / B <= g(pi_c) / g(pi*) <= (1 + eps) 2 / B, where pi* minimizes F_conn and g(pi) = sum_E w_E (|E| - lambda_pi(E))
# (Proposition 1). "We checked the instance by exhaustive enumeration for B = 3 and B = 4."
import argparse
import itertools
from fractions import Fraction

import numpy as np


def partitions(n, B):
    """All partitions of range(n) into blocks of exactly B records, as page arrays; each partition appears once, so
    layouts that differ only by a renaming of pages are counted once."""
    page = [-1] * n

    def rec(p):
        try:
            i = page.index(-1)
        except ValueError:
            yield list(page)
            return
        free = [j for j in range(i + 1, n) if page[j] == -1]
        # the smallest free record opens page p; choose its B - 1 page-mates
        for comb in itertools.combinations(free, B - 1):
            page[i] = p
            for j in comb:
                page[j] = p
            yield from rec(p + 1)
            page[i] = -1
            for j in comb:
                page[j] = -1
    yield from rec(0)


def layout_values(sets, layouts):
    """For every layout (row of layouts) and integer-weighted sets [(E, w)]: reads sum_E w_E lambda(E), the set credit
    g = sum_E w_E (|E| - lambda(E)) and the clique objective F_clq = sum_E w_E (pairs of E on one page)."""
    L = layouts.shape[0]
    R = np.zeros(L, np.int64); g = np.zeros(L, np.int64); F = np.zeros(L, np.int64)
    for E, w in sets:
        sub = np.sort(layouts[:, list(E)], axis=1)
        lam = 1 + (np.diff(sub, axis=1) != 0).sum(1)
        pairs = np.zeros(L, np.int64)
        for i, j in itertools.combinations(range(len(E)), 2):
            pairs += (layouts[:, E[i]] == layouts[:, E[j]])
        R += w * lam; g += w * (len(E) - lam); F += w * pairs
    return R, g, F


def check(B, eps):
    """Enumerate the grid instance for one B and eps; return the quantities that Proposition 3 names."""
    n = B * B
    rows = [tuple(i * B + j for j in range(B)) for i in range(B)]
    column_pairs = [(i * B + j, i2 * B + j) for j in range(B) for i in range(B) for i2 in range(i + 1, B)]
    # integer weights: eps = a / b gives rows a + b and column pairs b, i.e. the paper's weights times b
    a, b = eps.numerator, eps.denominator
    sets = [(r, a + b) for r in rows] + [(cp, b) for cp in column_pairs]
    layouts = np.array(list(partitions(n, B)), dtype=np.int8)
    R, g, F = layout_values(sets, layouts)
    clique_max = np.nonzero(F == F.max())[0]
    conn_min = np.nonzero(R == R.min())[0]   # F_conn = reads - sum_E w_E, so its minimizers are those of the reads
    row_layout = np.repeat(np.arange(B), B).astype(np.int8)
    g_c = Fraction(int(g[clique_max].min()), b)
    g_star = Fraction(int(g[conn_min[0]]), b)
    assert g_star == Fraction(int(g.max()), b)   # minimizing F_conn maximizes g
    unique_row = len(clique_max) == 1 and np.array_equal(layouts[clique_max[0]], row_layout)
    return dict(B=B, eps=eps, layouts=layouts.shape[0], clique_maximizers=len(clique_max),
                maximizer_is_row_layout=bool(unique_row),
                g_c=g_c, g_star=g_star, ratio=g_c / g_star, lower=Fraction(2, B), upper=(1 + eps) * Fraction(2, B))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', nargs='+', default=['3:1/9', '3:1/99', '4:1/9'], metavar='B:EPS',
                    help='page capacity B and weight excess eps > 0 of the rows, as a fraction; B = 4 enumerates '
                         '2,627,625 layouts')
    a = ap.parse_args()
    for case in a.cases:
        B, eps = case.split(':')
        r = check(int(B), Fraction(eps))
        print(f"B={r['B']} eps={r['eps']}: {r['layouts']} layouts; maximizers of F_clq: {r['clique_maximizers']} "
              f"(the row layout: {r['maximizer_is_row_layout']}); g(pi_c)={r['g_c']}, g(pi*)={r['g_star']}, "
              f"g(pi_c)/g(pi*)={r['ratio']} = {float(r['ratio']):.4f} in [2/B, (1+eps) 2/B] = "
              f"[{float(r['lower']):.4f}, {float(r['upper']):.4f}]: {r['lower'] <= r['ratio'] <= r['upper']}",
              flush=True)


if __name__ == '__main__':
    main()
