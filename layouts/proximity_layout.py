"""Proximity baseline: recursive balanced 2-means on the record vectors down to pages of B records, without logs."""
# [Paper] Section VII-A, Compared methods, "RCM and Proximity" (in the spirit of VeloANN); Table III, row Proximity.
import argparse
import json
import os
import time

import numpy as np
import numba as nb


@nb.njit(cache=True)
def bisect_all(X, n, d, B, seed):
    """Order of the records after recursive balanced 2-means splits; every leaf holds at most B records.

    Each split keeps a multiple of B records on the left, so every page but the last is full. A split seeds its two
    centres with two random members and runs 3 iterations that sort the members by their signed distance to the
    bisecting hyperplane and recompute the centres of the two halves."""
    np.random.seed(seed)
    perm = np.arange(n, dtype=np.int64)
    stack = [(0, n)]
    c1 = np.zeros(d, np.float32); c2 = np.zeros(d, np.float32)
    while len(stack) > 0:
        lo, hi = stack.pop(); L = hi - lo
        if L <= B:
            continue
        pages = (L + B - 1) // B; nl = ((pages + 1) // 2) * B
        seg = perm[lo:hi]
        a = seg[np.random.randint(0, L)]; b = seg[np.random.randint(0, L)]
        c1[:] = X[a]; c2[:] = X[b]
        score = np.zeros(L, np.float32)
        for it in range(3):
            dv = c1 - c2
            for i in range(L):
                x = X[seg[i]]; s = 0.0
                for j in range(d):
                    s += (x[j] - 0.5 * (c1[j] + c2[j])) * dv[j]
                # a small score means closer to c1
                score[i] = -s
            o = np.argsort(score, kind='mergesort'); seg[:] = seg[o]; score = score[o]
            c1[:] = 0; c2[:] = 0
            for i in range(nl):
                c1 += X[seg[i]]
            for i in range(nl, L):
                c2 += X[seg[i]]
            c1 /= nl; c2 /= (L - nl)
        stack.append((lo, lo + nl)); stack.append((lo + nl, hi))
    return perm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--vectors', required=True, help='record vectors: int32 n, int32 d, then n * d int8 values')
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--seed', type=int, default=0x13528ed, help='seed of the centre draws')
    ap.add_argument('--out', required=True, help='output layout (.npy); run details go to the same name with .json')
    a = ap.parse_args()

    t0 = time.time()
    hdr = np.fromfile(a.vectors, np.int32, 2); n, d = int(hdr[0]), int(hdr[1])
    assert os.path.getsize(a.vectors) == 8 + n * d, 'expected int8 vectors'
    X = np.memmap(a.vectors, dtype=np.int8, mode='r', offset=8, shape=(n, d)).astype(np.float32)
    perm = bisect_all(X, n, d, a.capacity, a.seed)
    page = np.zeros(n, np.int32)
    page[perm] = (np.arange(n) // a.capacity).astype(np.int32)
    assert np.bincount(page).max() <= a.capacity and len(np.unique(perm)) == n
    np.save(a.out, page)
    info = dict(layout='Proximity', capacity=a.capacity, records=n, pages=int(page.max()) + 1, seed=a.seed,
                seconds=time.time() - t0)
    json.dump(info, open(a.out[:-4] + '.json' if a.out.endswith('.npy') else a.out + '.json', 'w'), indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
