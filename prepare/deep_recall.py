"""Recall@10 of the logged DEEP1M test calls: 5,000 sampled test calls, top-10 by float64 squared L2 (the 11 nearest
from argpartition, ordered by distance, then identifier), compared with the results of the logging run."""
import argparse
import json
import os

import numpy as np

from hyperpage import formats

# [Paper] Table II, row DEEP1M, column recall@10


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True, help='DATA_DIR/deep: base.fbin, queries_needed.npy, splits.npz')
    ap.add_argument('--heldout', required=True, help='held-out DEEP rows as headerless float32 (see README)')
    ap.add_argument('--log-dir', required=True, help='output directory of tune_and_log.py (one run, all_c0)')
    ap.add_argument('--out', required=True, help='output JSON')
    ap.add_argument('--n-sample', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0x13528a7)
    a = ap.parse_args()
    D = a.data_dir
    with open(os.path.join(D, 'base.fbin'), 'rb') as f:
        n, d = np.fromfile(f, np.int32, 2)
        X = np.fromfile(f, np.float32, n * d).reshape(n, d).astype(np.float64)
    res = formats.read_results(os.path.join(a.log_dir, 'all_c0'))
    nq, k = res.shape
    need = np.load(os.path.join(D, 'queries_needed.npy'))
    assert len(need) == nq, (len(need), nq)
    calls = np.load(os.path.join(D, 'splits.npz'))['test_q']
    pick = np.sort(np.random.default_rng(a.seed).choice(len(calls), a.n_sample, replace=False))
    qid = calls[pick]
    pos = np.searchsorted(need, qid)
    assert (need[pos] == qid).all()
    raw = np.memmap(a.heldout, dtype=np.float32, mode='r').reshape(-1, d)
    Q = np.asarray(raw[qid], dtype=np.float64)
    xn = (X * X).sum(1)
    hit = 0
    for c in range(0, len(Q), 50):
        q = Q[c:c + 50]
        dist = xn[None, :] + (q * q).sum(1)[:, None] - 2.0 * q @ X.T
        part = np.argpartition(dist, 10, axis=1)[:, :11]
        for i in range(len(q)):
            # the 11 smallest candidates, ordered by (distance, identifier); the first 10 are the neighbours
            o = part[i][np.lexsort((part[i], dist[i, part[i]]))][:10]
            hit += len(set(res[pos[c + i], :10].tolist()) & set(o.tolist()))
    r = dict(split='deep_test', calls_in_split=int(len(calls)), sampled=int(len(Q)), res_k=int(k),
             recall10=hit / (len(Q) * 10), n=int(n), d=int(d))
    json.dump(r, open(a.out, 'w'), indent=1)
    print(json.dumps(r))


if __name__ == '__main__':
    main()
