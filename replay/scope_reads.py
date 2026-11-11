"""Mean page reads per group and per tool call under read-once scopes, and every layout's saving over the incumbent."""
# [Paper] Table VI: reads per W3 test group (keyed group buffer) and per tool call, Delta = 1 - R_A / R_MARGO; the same
# group means give the ablation steps of Section VII-C (arc restriction, set form, P05, SHP's move rule, shuffled
# groups).
import argparse
import json

import numpy as np

from _scopes import expand, group_reads, scope_reads, tool_call_scopes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--unions', required=True, help='group unions of the test split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--sets', required=True, help='per-query sets of the test split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--schedule', required=True, help='arrival metadata of the test split (make_schedule.py)')
    ap.add_argument('--layout', action='append', required=True, metavar='NAME=FILE',
                    help='layout; FILE may contain {draw} (one layout per draw); repeat for several layouts')
    ap.add_argument('--draws', default='0,1,2')
    ap.add_argument('--pages', type=int, required=True, help='pages of the index')
    ap.add_argument('--incumbent', default='MARGO')
    ap.add_argument('--out', default=None, help='optional JSON with the values')
    a = ap.parse_args()

    draws = [int(x) for x in a.draws.split(',')]
    u = np.load(a.unions); q = np.load(a.sets); s = np.load(a.schedule)
    u_ptr, u_nodes = u['e_ptr'].astype(np.int64), u['e_nodes'].astype(np.int32)
    q_ptr, q_nodes = q['e_ptr'].astype(np.int64), q['e_nodes'].astype(np.int32)
    n_steps = int(s['n_steps'])
    order, call, n_calls = tool_call_scopes(s)
    out = dict(groups=len(u_ptr) - 1, tool_calls=n_calls, steps=n_steps, layouts={})
    for spec in a.layout:
        name, files = expand(spec, draws)
        per = []
        for f in files:
            page = np.load(f).astype(np.int32)
            group = float(group_reads(u_ptr, u_nodes, page, a.pages).mean())
            per_call = float(scope_reads(order, q_ptr, q_nodes, call, n_calls, page, a.pages).sum() / n_steps)
            per.append(dict(group=group, tool_call_per_step=per_call))
        out['layouts'][name] = dict(files=files, per_draw=per, group=float(np.mean([x['group'] for x in per])),
                                    call=float(np.mean([x['tool_call_per_step'] for x in per])))
    base = out['layouts'].get(a.incumbent)
    print(f'{"layout":14s} {"group":>10s} {"Delta %":>8s} {"call Delta %":>13s}')
    for name, r in out['layouts'].items():
        if base:
            r['delta_group'] = 100 * (1 - r['group'] / base['group'])
            r['delta_call'] = 100 * (1 - r['call'] / base['call'])
            print(f'{name:14s} {r["group"]:10.1f} {r["delta_group"]:8.2f} {r["delta_call"]:13.2f}')
        else:
            print(f'{name:14s} {r["group"]:10.1f}')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
