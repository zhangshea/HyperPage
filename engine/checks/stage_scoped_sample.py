"""Stage the training groups on which the read-once scoped engine is checked: query file, scope pointers and, per
layout, the replayed reads lambda_pi(E_sigma) of every group union; the first groups also form the warm-up file."""
# [Paper] Section VII-A: "A patched DiskANN reproduces the stock results and expansion sequences of 7,850 W3
# sub-queries" -- 200 training groups of draw 1, chosen by a seeded permutation; no test group enters the engine here.
# The first 20 of them (799 sub-queries on W3) are the untimed warm-up of every timed run (Tables IV and V).
import argparse
import json
import os

import numpy as np


def write_bin(path, x):
    """DiskANN .bin matrix: int32 rows, int32 columns, then the rows."""
    with open(path, 'wb') as f:
        np.array(x.shape, np.int32).tofile(f); np.ascontiguousarray(x).tofile(f)


def distinct_pages(e_ptr, e_nodes, page, sets):
    """lambda_pi(E) = number of distinct pages of each listed set (Corollary 1: the reads of a read-once scope)."""
    out = np.zeros(len(sets), np.int64)
    for i, e in enumerate(sets):
        out[i] = len(np.unique(page[e_nodes[e_ptr[e]:e_ptr[e + 1]]]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--splits', required=True,
                    help='splits .npz with <key>_ptr (group pointers) and <key>_q (string ids)')
    ap.add_argument('--key', default='train1', help='split key of the training draw')
    ap.add_argument('--unions', required=True, help='group unions of the same split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--queries', required=True, help='int8 query vectors by string id (.npy)')
    ap.add_argument('--layout', action='append', default=[], metavar='NAME=FILE',
                    help='layout (.npy) to stage reads for')
    ap.add_argument('--groups', type=int, default=200, help='groups of the check sample')
    ap.add_argument('--warm-groups', type=int, default=20, help='first groups of the sample that form the warm-up file')
    ap.add_argument('--seed', type=int, default=0x13528f1, help='seed of the permutation of the training groups')
    ap.add_argument('--out', required=True, help='output directory')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    S = np.load(a.splits); ptr, q = S[f'{a.key}_ptr'], S[f'{a.key}_q']
    Q8 = np.load(a.queries, mmap_mode='r')
    ng = len(ptr) - 1
    groups = np.random.default_rng(a.seed).permutation(ng)[:a.groups]
    info = dict(seed=a.seed, groups_total=int(ng))

    def emit(name, ids):
        qq = np.concatenate([q[ptr[b]:ptr[b + 1]] for b in ids])
        sz = np.array([ptr[b + 1] - ptr[b] for b in ids], np.int64)
        bp = np.zeros(len(sz) + 1, np.uint64); bp[1:] = np.cumsum(sz)
        write_bin(f'{a.out}/{name}_q.bin', np.ascontiguousarray(Q8[qq])); bp.tofile(f'{a.out}/{name}_bptr.u64')
        np.save(f'{a.out}/{name}_strid.npy', qq); np.save(f'{a.out}/{name}_groups.npy', ids)
        info[name] = dict(groups=int(len(sz)), calls=int(sz.sum()))

    emit('scope', groups)
    emit('warm', groups[:a.warm_groups])

    U = np.load(a.unions); e_ptr, e_nodes = U['e_ptr'].astype(np.int64), U['e_nodes'].astype(np.int64)
    assert len(e_ptr) - 1 == ng, 'unions and splits disagree on the number of groups'
    info['lambda_mean'] = {}
    for item in a.layout:
        name, path = item.split('=', 1)
        page = np.load(path).astype(np.int64)
        lam = distinct_pages(e_ptr, e_nodes, page, groups)
        np.save(f'{a.out}/lambda_{name}.npy', lam); info['lambda_mean'][name] = float(lam.mean())
    json.dump(info, open(f'{a.out}/scope_info.json', 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
