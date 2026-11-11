"""Bootstrap over prompt groups of read-once group margins: 1 - sum_b R_A(b) / sum_b R_P(b) with resampled groups."""
# [Paper] Section VII-A: at 1M under the keyed group buffer, the margins of SET-G over MARGO and of the first three
# ablation steps (ARC-G over MARGO, CLQ-G over ARC-G, SET-G over CLQ-G) are positive at level 0.05 under a bootstrap
# over prompts. A margin is positive when the 5th percentile of its resampled values is above zero.
import argparse
import json

import numpy as np

from _scopes import expand, group_reads


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--unions', required=True, help='group unions of the test split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--layout', action='append', required=True, metavar='NAME=FILE', help='FILE may contain {draw}')
    ap.add_argument('--pair', action='append', required=True, metavar='A:P', help='margin of layout A over layout P')
    ap.add_argument('--draws', default='0,1,2', help='per-group reads are averaged over these draws')
    ap.add_argument('--pages', type=int, required=True)
    ap.add_argument('--resamples', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0x1352b58)
    ap.add_argument('--level', type=float, default=0.05)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    draws = [int(x) for x in a.draws.split(',')]
    u = np.load(a.unions)
    u_ptr, u_nodes = u['e_ptr'].astype(np.int64), u['e_nodes'].astype(np.int32)
    reads = {}
    for spec in a.layout:
        name, files = expand(spec, draws)
        reads[name] = np.mean([group_reads(u_ptr, u_nodes, np.load(f).astype(np.int32), a.pages).astype(float)
                               for f in files], axis=0)
    n = len(u_ptr) - 1
    # multinomial weights over the groups: one resample of the prompt groups per row, shared by every pair
    weights = np.random.default_rng(a.seed).multinomial(n, np.ones(n) / n, size=a.resamples).astype(float)
    out = {}
    for pair in a.pair:
        x, y = pair.split(':')
        boot = 1 - (weights @ reads[x]) / (weights @ reads[y])
        low = float(np.percentile(boot, 100 * a.level))
        out[pair] = dict(margin=float(1 - reads[x].sum() / reads[y].sum()), lower=low, positive=low > 0)
        print(f'{x} over {y}: margin {100 * out[pair]["margin"]:.2f}%, {100 * a.level:g}th percentile '
              f'{100 * low:.2f}%, positive at level {a.level}: {"yes" if low > 0 else "no"}')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
