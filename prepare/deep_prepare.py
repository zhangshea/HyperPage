"""DEEP1M inputs: the 1M base records, the validation queries and their top-100 by squared L2 for choosing L, and the
list of held-out rows (training pool, validation and test batches) that the logging run traverses."""
import argparse
import os

import numpy as np


def read_fbin_rows(p, n0, n1):
    with open(p, 'rb') as f:
        N, d = np.fromfile(f, dtype=np.int32, count=2)
        f.seek(8 + 4 * d * n0)
        x = np.fromfile(f, dtype=np.float32, count=(n1 - n0) * d)
    assert x.size == (n1 - n0) * d, (p, x.size)
    return x.reshape(n1 - n0, d)


def write_fbin(p, x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    with open(p, 'wb') as f:
        np.array(x.shape, dtype=np.int32).tofile(f)
        x.tofile(f)


def exact_topk_l2(base, q, k, bs=2000):
    """Top-k by squared L2 in float32 (||x||^2 - 2 q.x, the constant ||q||^2 omitted): the k nearest from argpartition,
    then sorted by distance with a stable sort, so equal distances keep their order in the partition."""
    ids = np.empty((q.shape[0], k), np.int64)
    bn = (base.astype(np.float32) ** 2).sum(1)
    for i in range(0, q.shape[0], bs):
        qq = q[i:i + bs]
        d = bn[None, :] - 2 * (qq @ base.T)
        part = np.argpartition(d, k, axis=1)[:, :k]
        pd = np.take_along_axis(d, part, 1)
        o = np.argsort(pd, axis=1, kind='stable')
        ids[i:i + bs] = np.take_along_axis(part, o, 1)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--head', required=True, help='rows 0..1,031,999 of DEEP base.10M.fbin, as an .fbin file')
    ap.add_argument('--out', required=True, help='DATA_DIR/deep')
    ap.add_argument('--n-base', type=int, default=1000000, help='base records: rows 0..n_base-1')
    ap.add_argument('--val-rows', type=int, nargs=2, default=[1030000, 1032000], help='validation queries for L')
    ap.add_argument('--gt-k', type=int, default=100)
    ap.add_argument('--pool', type=int, default=1200000, help='held-out rows 0..pool-1 form the training pool')
    ap.add_argument('--val', type=int, nargs=2, default=[1268000, 20000], help='first held-out row and rows')
    ap.add_argument('--test', type=int, nargs=2, default=[1368000, 200000], help='first held-out row and rows')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = read_fbin_rows(a.head, 0, a.n_base)
    write_fbin(os.path.join(a.out, 'base.fbin'), base)
    val = read_fbin_rows(a.head, *a.val_rows)
    write_fbin(os.path.join(a.out, 'gtq_val.bin'), val)
    np.save(os.path.join(a.out, 'gt_val.npy'), exact_topk_l2(base, val, a.gt_k))
    # held-out row k is row 1,032,000 + k of base.10M.fbin; these are the queries of the DEEP1M batches
    need = np.concatenate([np.arange(0, a.pool), np.arange(a.val[0], a.val[0] + a.val[1]),
                           np.arange(a.test[0], a.test[0] + a.test[1])]).astype(np.int64)
    np.save(os.path.join(a.out, 'queries_needed.npy'), need)
    print('base', base.shape, 'val', val.shape, 'needed', len(need), flush=True)


if __name__ == '__main__':
    main()
