"""Exact top-10 neighbours on the int8 records for sampled validation and test calls, by squared L2 distance with ties
broken by the smaller identifier; used to choose L and to report recall@10."""
import argparse
import os

import numpy as np
import numba as nb


def read_bin(p, dtype):
    with open(p, 'rb') as f:
        n, d = np.fromfile(f, np.int32, 2)
        return np.fromfile(f, dtype, n * d).reshape(n, d)


def write_bin(p, x):
    with open(p, 'wb') as f:
        np.array(x.shape, np.int32).tofile(f)
        x.tofile(f)


def sample_queries(D, n_sample, seeds):
    """Draws n_sample calls of each split without replacement and writes their int8 vectors (gtq_<split>.bin) and
    string ids (gtq_<split>_strid.npy)."""
    Q8 = np.load(os.path.join(D, 'qx8.npy'), mmap_mode='r')
    S = np.load(os.path.join(D, 'splits.npz'))
    for split, seed in seeds:
        calls = S[f'{split}_q']
        pick = np.sort(np.random.default_rng(seed).choice(len(calls), n_sample, replace=False))
        qid = calls[pick]
        write_bin(os.path.join(D, f'gtq_{split}.bin'), np.ascontiguousarray(Q8[qid]))
        np.save(os.path.join(D, f'gtq_{split}_strid.npy'), qid)


def topk_torch(X, q, device):
    """float64 arithmetic, so squared distances between int8 vectors are exact integers; key = dist * 2^20 + id is
    unique for n < 2^20 and its 10 smallest values give the neighbours with ties broken by identifier."""
    import torch
    n = X.shape[0]
    assert n < (1 << 20)
    Xg = torch.from_numpy(X.astype(np.float64)).to(device)
    xn = (Xg * Xg).sum(1)
    ids = torch.arange(n, device=device, dtype=torch.float64)
    gt = np.zeros((len(q), 10), np.int64)
    for c in range(0, len(q), 100):
        qg = torch.from_numpy(q[c:c + 100].astype(np.float64)).to(device)
        dist = xn[None, :] + (qg * qg).sum(1)[:, None] - 2.0 * qg @ Xg.T
        key = dist * 1048576.0 + ids[None, :]
        top = torch.topk(key, 10, dim=1, largest=False, sorted=True).values
        gt[c:c + 100] = (torch.remainder(top, 1048576.0)).long().cpu().numpy()
    return gt


@nb.njit(parallel=True, cache=True)
def topk_scan(X, Q):
    """Integer squared L2 by a full scan per query; any n."""
    n, d = X.shape
    nq = Q.shape[0]
    res = np.zeros((nq, 10), np.int64)
    for q in nb.prange(nq):
        y = Q[q].astype(np.int32)
        bd = np.full(10, np.iinfo(np.int64).max, np.int64)
        bi = np.full(10, -1, np.int64)
        for i in range(n):
            s = 0
            x = X[i]
            for j in range(d):
                t = int(x[j]) - y[j]
                s += t * t
            if s < bd[9]:      # strict: a later id never displaces an equal distance, so ties go to the smaller id
                k = 9
                while k > 0 and bd[k - 1] > s:
                    bd[k] = bd[k - 1]
                    bi[k] = bi[k - 1]
                    k -= 1
                bd[k] = s
                bi[k] = i
        res[q] = bi
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True, help='base.bin, and qx8.npy + splits.npz unless --no-sample')
    ap.add_argument('--backend', choices=['torch', 'scan'], default='torch',
                    help='torch: float64 matrix products (n < 2^20); scan: numba integer scan (any n)')
    ap.add_argument('--device', default='cpu', help="torch backend: 'cpu' or 'cuda'")
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--no-sample', action='store_true', help='reuse existing gtq_<split>.bin files')
    ap.add_argument('--n-sample', type=int, default=5000)
    ap.add_argument('--val-seed', type=int, default=0x13528a6)
    ap.add_argument('--test-seed', type=int, default=0x13528a7)
    a = ap.parse_args()
    D = a.data_dir
    seeds = (('val', a.val_seed), ('test', a.test_seed))
    if not a.no_sample:
        sample_queries(D, a.n_sample, seeds)
    X = read_bin(os.path.join(D, 'base.bin'), np.int8)
    if a.backend == 'scan':
        nb.set_num_threads(a.threads)
    elif a.device == 'cpu':
        import torch
        torch.set_num_threads(a.threads)
    for split, _ in seeds:
        q = read_bin(os.path.join(D, f'gtq_{split}.bin'), np.int8)
        gt = topk_torch(X, q, a.device) if a.backend == 'torch' else topk_scan(X, q)
        np.save(os.path.join(D, f'gt_{split}.npy'), gt)
        print(split, 'ground truth', gt.shape, flush=True)


if __name__ == '__main__':
    main()
