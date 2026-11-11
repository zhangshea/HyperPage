"""Record cache list of the page-search engine: the B * D records most frequent in the training unions (ties to the
smaller id), as u32 ids for the RECORD_CACHE_LIST input of the ported engine."""
# [Paper] Section VII-B, MARGO's page-search engine: "a record cache of the whole budget ranked by training frequency".
import argparse

import numpy as np

from hyperpage import cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('unions', help='training group unions (.npz with e_ptr, e_nodes)')
    ap.add_argument('budget', type=int, help='budget D in pages')
    ap.add_argument('records', type=int, help='records of the index, n')
    ap.add_argument('out', help='output list (.u32)')
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    a = ap.parse_args()
    z = np.load(a.unions)
    f = cache.record_frequency(z['e_ptr'].astype(np.int64), z['e_nodes'].astype(np.int32), a.records)
    lst = cache.rank_records(f)[:a.capacity * a.budget].astype(np.uint32); lst.tofile(a.out)
    print(f'cache list {a.out}: {len(lst)} records, min freq {f[lst[-1]]}, covered {(f > 0).sum()}')


if __name__ == '__main__':
    main()
