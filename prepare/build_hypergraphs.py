"""Hypergraph files of every split from the logged expanded sets: <key>_Q.npz (one set e_q per sub-query, in split
order) and <key>_U.npz (one scope union E per scope), plus hyper_info.json with their sizes."""
import argparse
import json
import os

import numpy as np

from hyperpage import formats, hypergraph


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True, help='splits.npz and queries_needed.npy of the workload')
    ap.add_argument('--log-dir', required=True, help='output directory of tune_and_log.py')
    ap.add_argument('--out', required=True, help='HYPER_DIR/<workload>')
    ap.add_argument('--keys', default=None,
                    help='comma-separated split keys to build (default: every key of splits.npz)')
    a = ap.parse_args()
    S = np.load(os.path.join(a.data_dir, 'splits.npz'))
    need = np.load(os.path.join(a.data_dir, 'queries_needed.npy'))
    CH = json.load(open(os.path.join(a.log_dir, 'tune_log_summary.json')))['chunk_size']
    nchunk = -(-len(need) // CH)
    logs = [formats.read_expanded_sets(os.path.join(a.log_dir, f'all_c{c}')) for c in range(nchunk)]
    # position of every needed string id in the logging order
    pos_of = np.full(int(need.max()) + 1, -1, np.int64)
    pos_of[need] = np.arange(len(need))
    # one CSR of expanded sets over the logged strings, in logging order
    lens = np.concatenate([np.diff(L[0]) for L in logs])
    qptr = np.zeros(len(lens) + 1, np.int64)
    qptr[1:] = np.cumsum(lens)
    qnodes = np.concatenate([L[1] for L in logs]).astype(np.int32)
    del logs
    assert len(lens) == len(need)
    n = int(qnodes.max()) + 1
    os.makedirs(a.out, exist_ok=True)

    keys = sorted({k[:-4] for k in S.files if k.endswith('_ptr')})
    if a.keys:
        keys = [k for k in keys if k in a.keys.split(',')]
    info_path = os.path.join(a.out, 'hyper_info.json')
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    for key in keys:
        sptr = S[key + '_ptr']
        calls = pos_of[S[key + '_q']]
        assert (calls >= 0).all(), f'{key}: a sub-query string was not logged'
        # [Paper] Definition 8: the scope union is the union of the expanded sets of the scope's sub-queries
        eu, nu = hypergraph.scope_unions(sptr, calls, qptr, qnodes, n)
        np.savez(os.path.join(a.out, f'{key}_U.npz'), e_ptr=eu, e_nodes=nu)
        if key.endswith('_shuf'):
            # a shuffled split holds the same sub-queries as its source split, so its per-query sets are not stored
            eq, nq_ = np.zeros(1, np.int64), np.zeros(0, np.int32)
        else:
            eq, nq_ = hypergraph.per_query_sets(calls, qptr, qnodes)
            np.savez(os.path.join(a.out, f'{key}_Q.npz'), e_ptr=eq, e_nodes=nq_, batch_ptr=sptr)
        info[key] = dict(batches=int(len(sptr) - 1), calls=int(len(calls)), nnz_U=int(len(nu)), nnz_Q=int(len(nq_)),
                         mean_U=float(len(nu) / max(1, len(sptr) - 1)), mean_e_q=float(len(nq_) / max(1, len(calls))))
        del eu, nu, eq, nq_
        print(key, info[key], flush=True)
    json.dump(info, open(info_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
