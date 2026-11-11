"""Summary of the small-instance runs: instances solved to proven optimality and the ratio of the one-class-per-record
bound of Proposition 7 to the optimum, per (records, B)."""
# [Paper] Section VII-D: 71 instances solved to proven optimality; median bound/optimum 96% at B = 3, 81% to 90% at
# B = 12.
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+', help='output JSON files of small_instances.py, one per (records, B)')
    A = ap.parse_args()
    rows = []; total = 0
    for f in A.runs:
        d = json.load(open(f)); I = d['instances']
        for x in I:
            # without a CP-SAT solution (no opt_recheck), 'opt' holds no layout's reads: no check, and the instance
            # counts as unsolved unless the matching below proves it
            if 'opt_recheck' in x:
                assert x['lb_single']['value'] <= x['opt'] + 1e-6, f
                assert abs(x['opt'] - x['opt_recheck']) < 1e-6
            elif x['opt_status'] == 'OPTIMAL':
                x['opt_status'] = 'NO_SOLUTION'
        for x in I:
            # B = 2: the maximum-weight perfect matching (Corollary 2) is an exact optimum, so the instance is proven
            if 'opt_matching' in x:
                if 'opt_recheck' in x:
                    assert x['opt_matching'] <= x['opt'] + 1e-6
                assert abs(x['opt_matching'] - x['opt_matching_recheck']) < 1e-6
                assert x['lb_single']['value'] <= x['opt_matching'] + 1e-6
                x['opt'], x['opt_status'] = x['opt_matching'], 'OPTIMAL'
        opt = [x for x in I if x['opt_status'] == 'OPTIMAL']
        ratio = [x['lb_single']['value'] / x['opt'] for x in opt]
        rows.append((d['B'], d['n'], len(I), len(opt), ratio)); total += len(opt)
    print('| records | B | instances | proven | bound/optimum median [min, max] |')
    print('|---|---|---|---|---|')
    for B, n, ni, no, ratio in sorted(rows):
        cell = f'{np.median(ratio):.3f} [{min(ratio):.3f}, {max(ratio):.3f}]' if ratio else '--'
        print(f'| {n} | {B} | {ni} | {no} | {cell} |')
    print('proven in total:', total)


if __name__ == '__main__':
    main()
