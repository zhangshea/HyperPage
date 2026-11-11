"""Coverage of the record cache: shares of the records, of the test accesses and of the training incidences it holds."""
# [Paper] Section VII-B: at their validation shares the record caches of 15% and 25% of the full corpus hold 11.2% and
# 22.5% of the records, serve 74.8% and 84.7% of the test accesses and carry 79% and 92% of the training incidences;
# the training unions cover 33% of the full corpus's records and 86% of the 1M index's (Scale).
# [Paper] Introduction: 1% of the records serve 24.6% of the test accesses at 1M and 36.4% at 21M (--capacity 1 and
# --cell round(0.01 n):1, which caches exactly round(0.01 n) records).
import argparse
import json

import numpy as np

from hyperpage import cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='per-query sets of the test split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--train-unions', required=True, help='training group unions; {draw} is replaced by the draw')
    ap.add_argument('--records', type=int, required=True, help='records of the index')
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--cell', action='append', default=[], metavar='PAGES:SHARE',
                    help='budget in pages and hot-record share (the incumbent\'s validation share); repeatable')
    ap.add_argument('--draws', default='0,1,2')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    n = a.records
    q = np.load(a.sets)
    # accesses per record: occurrences over all test sub-queries, sum_q |e_q|
    accesses = np.bincount(q['e_nodes'].astype(np.int64), minlength=n).astype(np.float64)
    total = accesses.sum()
    cells = [(int(c.split(':')[0]), float(c.split(':')[1])) for c in a.cell]
    rows = {c: [] for c in cells}
    coverage = []
    for k in [int(x) for x in a.draws.split(',')]:
        z = np.load(a.train_unions.format(draw=k))
        incidences = z['e_nodes'].astype(np.int64)
        freq = cache.record_frequency(z['e_ptr'].astype(np.int64), z['e_nodes'].astype(np.int32), n)
        ranking = cache.rank_records(freq)
        coverage.append(float((freq > 0).mean()))
        for budget, share in cells:
            _, cached, _ = cache.split_budget(budget, share, a.capacity)
            hot = np.zeros(n, bool); hot[ranking[:cached]] = True
            rows[(budget, share)].append(dict(records=cached, share_of_records=cached / n,
                                              share_of_test_accesses=float(accesses[hot].sum() / total),
                                              share_of_training_incidences=float(hot[incidences].mean())))
    out = dict(training_coverage_of_records=float(np.mean(coverage)), per_draw_coverage=coverage, cells={})
    print(f'training unions cover {100 * out["training_coverage_of_records"]:.1f}% of the records (mean over draws)')
    for (budget, share), r in rows.items():
        m = {key: float(np.mean([x[key] for x in r])) for key in r[0]}
        out['cells'][f'{budget}|{share}'] = dict(mean=m, per_draw=r)
        print(f'budget {budget} pages, share {share}: {100 * m["share_of_records"]:.1f}% of the records, '
              f'{100 * m["share_of_test_accesses"]:.1f}% of the test accesses, '
              f'{100 * m["share_of_training_incidences"]:.1f}% of the training incidences')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
