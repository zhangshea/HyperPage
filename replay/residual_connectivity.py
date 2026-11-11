"""Connectivity of layouts on residual training unions: per group union, the pages of its uncached records, summed."""
# [Paper] Section VII-B: O_d(pi; h) = sum_b lambda_pi(E_b \ C_d(h)), where C_d(h) holds the B * h records with the
# most training incidences of draw d. It ranks the validation reads in rank_correlation.py, and its ratios give the
# connectivity comparisons of Mt-KaHyPar against MARGO and of P05-C against the two-level partitioner.
import argparse
import json

import numpy as np

from hyperpage import cache, hypergraph, objectives


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-unions', required=True, help='training group unions; {draw} is replaced by the draw')
    ap.add_argument('--layout', action='append', required=True, metavar='NAME=FILE',
                    help='FILE may contain {draw}; a FILE without {draw} serves every draw')
    ap.add_argument('--hot-pages', type=int, required=True,
                    help='h, the hot page slots of the column (floor of the incumbent\'s validation share times D)')
    ap.add_argument('--records', type=int, required=True)
    ap.add_argument('--pages', type=int, required=True)
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--draws', default='0,1,2')
    ap.add_argument('--out', required=True, help='JSON: per layout and draw the connectivity at h and at 0')
    a = ap.parse_args()

    n = a.records
    layouts = dict(spec.split('=', 1) for spec in a.layout)
    out = dict(hot_pages=a.hot_pages, capacity=a.capacity, rows={})
    for k in [int(x) for x in a.draws.split(',')]:
        z = np.load(a.train_unions.format(draw=k))
        e_ptr, e_nodes = z['e_ptr'].astype(np.int64), z['e_nodes'].astype(np.int32)
        ranking = cache.rank_records(cache.record_frequency(e_ptr, e_nodes, n))
        sets = {}
        for h in sorted({0, a.hot_pages}):
            cached = np.zeros(n, np.bool_); cached[ranking[:a.capacity * h]] = True
            sets[h] = hypergraph.residual_sets(e_ptr, e_nodes, cached)
        for name, pattern in layouts.items():
            page = np.load(pattern.format(draw=k)).astype(np.int32)
            assert len(page) == n and page.max() < a.pages, (name, len(page), int(page.max()))
            row = {f'h{h}': int(objectives.connectivity_and_clique(p, v, page, a.pages)[0])
                   for h, (p, v) in sets.items()}
            out['rows'][f'{name}|{k}'] = row
            print(name, k, row, flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
