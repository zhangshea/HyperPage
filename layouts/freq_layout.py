"""FREQ baseline: records packed into pages by their frequency in the training unions, B records per page."""
# [Paper] Section VII-A, Compared methods, "FREQ"; Table III, row FREQ; Table VI, row FREQ.
import argparse
import json

import numpy as np

from hyperpage.formats import layout_of_order


def freq_layout(e_ptr, e_nodes, start, capacity):
    """Layout that packs records by decreasing training frequency.

    Ties (in particular all records outside the training unions) keep the page order of the start layout, and records
    of one start page keep identifier order."""
    n = len(start)
    assert int(e_nodes.max()) < n, 'the training sets name a record outside the start layout'
    freq = np.bincount(np.asarray(e_nodes, np.int64), minlength=n).astype(np.int64)
    # one int64 key per record: start page first, identifier second
    tie = start[:n].astype(np.int64) * n + np.arange(n)
    order = np.lexsort((tie, -freq))
    return layout_of_order(order, n, capacity), freq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='training group unions of one draw (.npz with e_ptr, e_nodes)')
    ap.add_argument('--start', required=True, help="layout whose page order breaks ties (.npy); the paper uses MARGO's")
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--out', required=True, help='output layout (.npy); run details go to the same name with .json')
    a = ap.parse_args()

    z = np.load(a.sets)
    start = np.load(a.start)
    page, freq = freq_layout(z['e_ptr'], z['e_nodes'], start, a.capacity)
    n_pages = -(-len(page) // a.capacity)
    assert np.bincount(page, minlength=n_pages).max() <= a.capacity
    np.save(a.out, page)
    info = dict(layout='FREQ', capacity=a.capacity, records=int(len(page)), pages=int(n_pages),
                covered_records=int((freq > 0).sum()))
    json.dump(info, open(a.out[:-4] + '.json' if a.out.endswith('.npy') else a.out + '.json', 'w'), indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
