"""Memory of a keyed group buffer: pages held by all groups of a step, the sum of their read-once group reads."""
# [Paper] Definition 6 (keyed buffer) and Section VII-B, Read-once scopes: the live groups of a step hold 144,703
# pages at 1M (SET-G, three draws) and 2.54 to 2.95 GiB on the full corpus (draw-0 layouts).
import argparse
import json

import numpy as np

from _scopes import expand, group_reads


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--unions', required=True, help='group unions of the test split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--schedule', required=True, help='arrival metadata of the test split (make_schedule.py)')
    ap.add_argument('--layout', action='append', required=True, metavar='NAME=FILE', help='FILE may contain {draw}')
    ap.add_argument('--draws', default='0,1,2', help='draws averaged (the full-corpus range uses draw 0 only)')
    ap.add_argument('--pages', type=int, required=True, help='pages of the index')
    ap.add_argument('--page-bytes', type=int, default=8192)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    draws = [int(x) for x in a.draws.split(',')]
    u = np.load(a.unions); s = np.load(a.schedule)
    u_ptr, u_nodes = u['e_ptr'].astype(np.int64), u['e_nodes'].astype(np.int32)
    n_steps = int(s['n_steps'])
    # the step of every group, from the step of its sub-queries
    group_step = np.zeros(len(u_ptr) - 1, np.int64)
    group_step[s['group']] = s['step']
    out = {}
    for spec in a.layout:
        name, files = expand(spec, draws)
        per_step = [np.bincount(group_step, weights=group_reads(u_ptr, u_nodes, np.load(f).astype(np.int32), a.pages),
                                minlength=n_steps) for f in files]
        pages = float(np.mean([x.mean() for x in per_step]))
        out[name] = dict(pages_per_step=pages, pages_per_step_max=float(np.mean([x.max() for x in per_step])),
                         GiB_per_step=pages * a.page_bytes / 2 ** 30,
                         groups_per_step=float(np.bincount(group_step, minlength=n_steps).mean()))
        print(f'{name:14s} {pages:12,.2f} pages per step ({out[name]["GiB_per_step"]:.3f} GiB), '
              f'{out[name]["groups_per_step"]:.1f} groups per step')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
