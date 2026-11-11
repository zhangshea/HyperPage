"""Capacity bound L_cap = sum_E ceil(|E| / B) of Proposition 6 on unit-weight sets, reported per set.

Reads sets (e_ptr, e_nodes), for example the read-once test group unions, and prints and writes L_cap / m per page
capacity B."""
# [Paper] Proposition 6 (Capacity guarantee) and Section VII-D: L_cap = 193.1 reads per W3-1M test group at B = 12.
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='.npz with e_ptr, e_nodes')
    ap.add_argument('--out', required=True, help='output JSON')
    ap.add_argument('--B', default='12', help='comma-separated page capacities (records per page)')
    A = ap.parse_args()
    e_ptr = np.load(A.sets)['e_ptr'].astype(np.int64)
    sz = np.diff(e_ptr); m = len(sz)
    R = dict(sets=m)
    for B in (int(x) for x in A.B.split(',')):
        # every set of |E| records occupies at least ceil(|E| / B) pages under any layout with B records per page
        R[f'B{B}'] = dict(L_cap=float(np.ceil(sz / B).sum()), L_cap_per_set=float(np.ceil(sz / B).sum()) / m)
    json.dump(R, open(A.out, 'w'), indent=1); print(json.dumps(R))


if __name__ == '__main__':
    main()
