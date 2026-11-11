"""Replay the sub-queries of one split under the hybrid cache for a set of layouts, budgets and hot-record shares."""
# [Paper] Definition 7 and Section VII-A, Metrics: page reads per step under H_{D,rho}, kept across the steps of the
# split from an empty page LRU. The output feeds Tables III and VII and Figs. 3 and 4.
import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from hyperpage import cache

G = {}   # arrays shared with the worker processes


def ranking_of(draw):
    """Record frequency over the training group unions of one draw, and the admission order of the record cache."""
    if draw not in G['rank']:
        z = np.load(G['unions'].format(draw=draw))
        freq = cache.record_frequency(z['e_ptr'].astype(np.int64), z['e_nodes'].astype(np.int32), G['n'])
        G['rank'][draw] = (cache.rank_records(freq), freq)
    return G['rank'][draw]


def replay_one(task):
    name, draw, budget, share = task
    layout = np.load(G['layouts'][name].format(draw=draw)).astype(np.int32)
    hot_pages, cached_records, lru_pages = cache.split_budget(budget, share, G['capacity'], G['record_scale'])
    ranking, freq = ranking_of(draw)
    mask = cache.cached_mask(layout, G['pages'], ranking, freq, hot_pages, cached_records, G['cache_pages'])
    misses = cache.hybrid_cache_misses(G['order'], G['e_ptr'], G['e_nodes'], layout, G['pages'], lru_pages, mask,
                                       G['step'], G['n_steps'])
    return task, dict(per_step=misses.astype(float).tolist(), records=cached_records, lru_pages=lru_pages)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='per-query sets of the split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--schedule', required=True, help='arrival metadata of the split (.npz from make_schedule.py)')
    ap.add_argument('--train-unions', required=True,
                    help='training group unions that rank the hot records; {draw} is replaced by the draw index')
    ap.add_argument('--layout', action='append', required=True, metavar='NAME=FILE',
                    help='layout to replay; FILE may contain {draw}; repeat for several layouts')
    ap.add_argument('--draws', default='0,1,2', help='training draws')
    ap.add_argument('--pages', type=int, required=True, help='pages of the index, ceil(records / capacity)')
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--budgets', required=True, help='declared budgets in pages, comma separated')
    ap.add_argument('--shares', default=','.join(str(s) for s in cache.SHARE_GRID), help='hot-record shares rho')
    ap.add_argument('--at', default=None,
                    help='output of select_deployment.py, or a JSON map "layout|budget" -> share(s): '
                         'replay each layout at these shares only')
    ap.add_argument('--cache-pages', action='store_true',
                    help='static part of the cache holds the hottest pages instead of records')
    ap.add_argument('--record-scale', type=float, default=1.0,
                    help='memory of a cached record relative to its on-disk size')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    q = np.load(a.sets)
    s = np.load(a.schedule)
    G.update(e_ptr=q['e_ptr'].astype(np.int64), e_nodes=q['e_nodes'].astype(np.int32), step=s['step'].astype(np.int64),
             n_steps=int(s['n_steps']), pages=a.pages, capacity=a.capacity, record_scale=a.record_scale,
             cache_pages=a.cache_pages, unions=a.train_unions, rank={},
             layouts=dict(item.split('=', 1) for item in a.layout))
    assert len(G['step']) == len(G['e_ptr']) - 1, 'the schedule does not match the sets'
    G['order'] = cache.lockstep_order(s['step'], s['turn'], s['rollout'], s['position'])
    draws = [int(x) for x in a.draws.split(',')]
    budgets = [int(x) for x in a.budgets.split(',')]
    shares = [float(x) for x in a.shares.split(',')]
    first = next(iter(G['layouts'].values())).format(draw=draws[0])
    G['n'] = len(np.load(first, mmap_mode='r'))

    at = json.load(open(a.at)) if a.at else None
    if at is not None:
        at = at.get('test_shares', at.get('shares', at))   # output of select_deployment.py, or a plain map
    tasks = []
    for name in G['layouts']:
        for budget in budgets:
            for share in (np.atleast_1d(at[f'{name}|{budget}']).astype(float).tolist() if at else shares):
                tasks += [(name, k, budget, share) for k in draws]
    tasks = sorted(set(tasks))

    t0 = time.time()
    result = dict(sets=a.sets, pages=a.pages, capacity=a.capacity, budgets=budgets, shares=shares, draws=draws,
                  n_steps=G['n_steps'], rows={})
    mp.set_start_method('fork', force=True)
    with mp.Pool(a.workers) as pool:
        for task, row in pool.imap_unordered(replay_one, tasks):
            result['rows']['|'.join(str(x) for x in task)] = row
    result['seconds'] = time.time() - t0
    json.dump(result, open(a.out, 'w'))
    print(f'{len(tasks)} replays in {result["seconds"]:.0f} s -> {a.out}')


if __name__ == '__main__':
    main()
