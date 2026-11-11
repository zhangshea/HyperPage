"""Arrival order logged by the hybrid-cache engine as replay input: one set per search round, in log order."""
# [Paper] Section VII-B, Search time: the replay in the arrival order that the engine logs. The engine's order log
# (engine/patches/add_order_log.py) holds one record (sequence, call, round of the call, nodes expanded in the round)
# per search round; round t of a sub-query covers the next n_t records of its expanded set. Sorted by sequence, the
# rounds of all threads give the order of the LRU operations. The outputs are a sets file and a schedule for
# replay_hybrid.py, whose turn lockstep then visits the rounds in log order.
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='per-query sets of the split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--schedule', required=True, help='arrival metadata of the split (.npz from make_schedule.py)')
    ap.add_argument('--order-log', required=True, help='<out>.order of hybrid_search: u64 quadruples, one per round')
    ap.add_argument('--call-map', required=True,
                    help='engine call i is sub-query map[i] of the split '
                         '(<key>_lock.npy of engine/checks/stage_hybrid.py)')
    ap.add_argument('--out-sets', required=True, help='output .npz: one set per logged round, in log order')
    ap.add_argument('--out-schedule', required=True, help='output .npz: schedule of the rounds for replay_hybrid.py')
    a = ap.parse_args()

    q = np.load(a.sets)
    e_ptr, e_nodes = q['e_ptr'].astype(np.int64), q['e_nodes'].astype(np.int32)
    s = np.load(a.schedule)
    nc = len(e_ptr) - 1
    assert len(s['step']) == nc, 'the schedule does not match the sets'
    lock = np.load(a.call_map).astype(np.int64)
    assert len(lock) == nc, 'the call map does not match the sets'
    log = np.fromfile(a.order_log, dtype=np.uint64).reshape(-1, 4).astype(np.int64)
    seq, eng, rnd, cnt = log[:, 0], log[:, 1], log[:, 2], log[:, 3]
    call = lock[eng]

    # first record of every round inside its sub-query's set: the rounds of a sub-query in round order, cumulated
    o = np.lexsort((rnd, call))
    start = np.zeros(len(log), np.int64)
    cs = np.cumsum(cnt[o])
    first = np.r_[True, call[o][1:] != call[o][:-1]]
    base = np.maximum.accumulate(np.where(first, cs - cnt[o], 0))
    start[o] = cs - cnt[o] - base

    # integrity of the log: every sub-query logged, its rounds cover its set exactly, sequence numbers distinct
    tot = np.bincount(call, weights=cnt, minlength=nc).astype(np.int64)
    check = dict(records=int(len(log)), calls_logged=int(len(np.unique(call))), calls=int(nc),
                 calls_len_match=int((tot == np.diff(e_ptr)).sum()), seq_unique=bool(len(np.unique(seq)) == len(seq)))
    print('check', check)
    assert check['calls_logged'] == check['calls_len_match'] == nc and check['seq_unique'], 'incomplete order log'

    # the rounds in log order, each with the records it expands
    ev = np.argsort(seq, kind='stable')
    ev_call, ev_start, ev_n = call[ev], start[ev], cnt[ev]
    ev_ptr = np.zeros(len(ev) + 1, np.int64)
    ev_ptr[1:] = np.cumsum(ev_n)
    index = np.repeat(e_ptr[ev_call] + ev_start - ev_ptr[:-1], ev_n) + np.arange(ev_ptr[-1], dtype=np.int64)
    ev_nodes = e_nodes[index]

    # every round is charged to the step of its sub-query; waves are separated by a barrier, so steps never decrease
    step = s['step'].astype(np.int64)[ev_call]
    assert (np.diff(step) >= 0).all(), 'steps decrease along the log; turn lockstep cannot keep its order'
    n = len(ev)
    zeros = np.zeros(n, np.int64)
    np.savez(a.out_sets, e_ptr=ev_ptr, e_nodes=ev_nodes, call=ev_call)
    np.savez(a.out_schedule, n_steps=s['n_steps'], step=step, turn=zeros, rollout=zeros,
             position=np.arange(n, dtype=np.int64), group=s['group'].astype(np.int64)[ev_call],
             string=s['string'].astype(np.int64)[ev_call])
    print(f'{n} rounds of {nc} sub-queries, {int(s["n_steps"])} steps -> {a.out_sets}, {a.out_schedule}')


if __name__ == '__main__':
    main()
