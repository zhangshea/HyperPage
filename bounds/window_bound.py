"""Window bound of Proposition 8 on the page misses of every layout under the hybrid cache, per test step.

For each training draw (hot-record ranking), budget and hot-record share rho, the test sub-queries are taken in turn
lockstep and cut into consecutive windows; the bound is maximised over a grid of window sizes."""
# [Paper] Proposition 8 (Window bound) and Section VII-D: under the shared cache of Table III the bound lies 11 to 48
# times below HyperPage's reads (ratio = HyperPage's test reads per step / minimum bound per step over draws and rho).
#
# Budget of D pages, share rho: h = floor(rho D) hot pages hold the C = B h records most frequent in training
# (Definition 7) and the page LRU holds H = D - h pages. Window j collects the distinct uncached records S_j of
# consecutive sub-queries until |S_j| reaches T; every layout misses at least ceil(|S_1|/B) +
# sum_{j >= 2} max(0, ceil(|S_j|/B) - H) times. Each T gives a valid bound, so the largest over the grid is kept.
import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import numba as nb

from hyperpage.cache import lockstep_order
from hyperpage.hypergraph import most_frequent_records

WINDOW_FACTORS = (1.05, 1.1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16)
_S = {}   # inputs shared with the worker processes (fork)


@nb.njit(cache=True)
def window_bound(order, e_ptr, e_nodes, cached, stamp, T, H, B):
    """ceil(|S_1|/B) + sum_{j >= 2} max(0, ceil(|S_j|/B) - H) over consecutive windows; a window closes once it holds T
    distinct uncached records. stamp must be zero on entry. Returns the bound and the number of windows."""
    # [Paper] Proposition 8
    w = 1; cnt = 0; lb = 0; first = True; nwin = 0
    for i in range(len(order)):
        q = order[i]
        for j in range(e_ptr[q], e_ptr[q + 1]):
            v = e_nodes[j]
            if cached[v] == 0 and stamp[v] != w:
                stamp[v] = w; cnt += 1
        if cnt >= T:
            lam = (cnt + B - 1) // B
            if first:
                lb += lam; first = False
            elif lam > H:
                lb += lam - H
            nwin += 1; w += 1; cnt = 0
    if cnt > 0:
        lam = (cnt + B - 1) // B
        if first:
            lb += lam
        elif lam > H:
            lb += lam - H
        nwin += 1
    return lb, nwin


def hot_records(draw, count):
    """Mask of the count records most frequent in the training unions of a draw (ties by smaller identifier)."""
    # [Paper] Definition 7: the record cache holds the records most frequent in training
    key = ('unions', draw)
    if key not in _S:   # one load per draw and worker process
        z = np.load(_S['train_unions'][draw]); _S[key] = (z['e_ptr'].astype(np.int64), z['e_nodes'].astype(np.int64))
    tp, tn = _S[key]
    return most_frequent_records(tp, tn, _S['records'], count).astype(np.uint8)


def one_row(task):
    draw, bud, rho = task
    S = _S; B = S['B']; n = S['records']
    cap = S['budgets'][bud]; h = int(np.floor(rho * cap)); C = B * h; H = cap - h
    cached = hot_records(draw, C)
    best = (0, 0, 0); stamp = np.zeros(n, np.int32)
    Ts = sorted({int(x) for x in [B * max(H, 1) * f for f in WINDOW_FACTORS]})
    for T in Ts:
        stamp[:] = 0
        lb, nw = window_bound(S['order'], S['e_ptr'], S['e_nodes'], cached, stamp, T, H, B)
        if lb > best[0]:
            best = (int(lb), T, int(nw))
    return task, dict(lb_per_step=best[0] / S['steps'], T=best[1], windows=best[2], H=H, hot=C)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='per-query sets of the evaluation split (.npz: e_ptr, e_nodes)')
    ap.add_argument('--schedule', required=True,
                    help='arrival metadata of the split (.npz from replay/make_schedule.py)')
    ap.add_argument('--train-unions', nargs='+', required=True,
                    help='training group unions (.npz), one per draw, that rank the hot records')
    ap.add_argument('--out', required=True, help='output JSON')
    ap.add_argument('--records', type=int, required=True, help='records of the index (1000000 or 21015324)')
    ap.add_argument('--B', type=int, default=12, help='records per page')
    ap.add_argument('--budgets', required=True, help='declared budgets in pages, comma separated')
    ap.add_argument('--rho', default='0,0.1,0.25,0.5,0.75,0.9,1', help='grid of hot-record shares')
    ap.add_argument('--procs', type=int, default=1, help='worker processes')
    ap.add_argument('--reads-per-step', default='',
                    help='optional BUDGET=READS,... of a layout (test reads per step); prints reads / bound')
    A = ap.parse_args()
    t0 = time.time()
    buds = A.budgets.split(','); rhos = tuple(float(x) for x in A.rho.split(','))
    all_budgets = {b: int(b) for b in buds}
    Q = np.load(A.sets)
    sched = np.load(A.schedule)
    assert len(sched['step']) == len(Q['e_ptr']) - 1, 'the schedule does not match the sets'
    _S.update(B=A.B, records=A.records, budgets=all_budgets, train_unions=A.train_unions,
              steps=int(sched['n_steps']),
              e_ptr=Q['e_ptr'].astype(np.int64), e_nodes=Q['e_nodes'].astype(np.int32),
              order=lockstep_order(sched['step'], sched['turn'], sched['rollout'], sched['position']))
    tasks = sorted({(k, bud, rho) for k in range(len(A.train_unions)) for bud in buds for rho in rhos})
    R = dict(sets=A.sets, budgets_pages=all_budgets, rows={})
    mp.set_start_method('fork', force=True)
    with mp.Pool(A.procs) as pool:
        for t, row in pool.imap_unordered(one_row, tasks):
            R['rows']['|'.join(map(str, t))] = row; print(t, row, flush=True)
    # the reported ratio divides by the smallest bound over draws and shares of a budget
    R['min_lb_per_step'] = {b: min(v['lb_per_step'] for key, v in R['rows'].items() if key.split('|')[1] == b)
                            for b in buds}
    if A.reads_per_step:
        reads = {kv.split('=')[0]: float(kv.split('=')[1]) for kv in A.reads_per_step.split(',')}
        R['ratio'] = {b: reads[b] / R['min_lb_per_step'][b] for b in reads}
        print('reads / bound:', R['ratio'])
    R['secs'] = time.time() - t0; json.dump(R, open(A.out, 'w'), indent=1)
    print('min bound per step:', R['min_lb_per_step'])


if __name__ == '__main__':
    main()
