"""Replay under the hybrid cache behind an exact-string result cache that answers repeated sub-queries."""
# [Paper] Section VII-C: an exact-string result cache of a share phi of the budget, in front of the hybrid cache. Its
# entries take 72 bytes each, its pages are taken from the budget before the share split, and a hit skips the search.
import argparse
import json
import multiprocessing as mp
import time
from collections import OrderedDict

import numpy as np

from hyperpage import cache

G = {}   # arrays shared with the worker processes


def result_cache_hits(order, string, entries):
    """Hit mask of an LRU of `entries` query strings, visited in arrival order, starting empty."""
    hit = np.zeros(len(string), np.bool_)
    if entries <= 0:
        return hit
    lru = OrderedDict()
    for c in order:
        i = int(string[c])
        if i in lru:
            hit[c] = True; lru.move_to_end(i)
        else:
            lru[i] = None
            if len(lru) > entries:
                lru.popitem(last=False)
    return hit


def ranking_of(draw):
    if draw not in G['rank']:
        z = np.load(G['unions'].format(draw=draw))
        G['rank'][draw] = cache.rank_records(cache.record_frequency(z['e_ptr'].astype(np.int64),
                                                                    z['e_nodes'].astype(np.int32), G['n']))
    return G['rank'][draw]


def replay_one(task):
    name, draw, budget, share = task
    entries, rc_pages, hit = G['rc'][budget]
    layout = np.load(G['layouts'][name].format(draw=draw)).astype(np.int32)
    # the result cache's pages come out of the budget before the record cache and the page LRU split the rest
    hot_pages, cached_records, lru_pages = cache.split_budget(budget - rc_pages, share, G['capacity'])
    mask = np.zeros(G['n'], np.bool_)
    if cached_records > 0:
        mask[ranking_of(draw)[:cached_records]] = True
    order = G['order'][~hit[G['order']]]
    misses = cache.hybrid_cache_misses(order, G['e_ptr'], G['e_nodes'], layout, G['pages'], lru_pages, mask, G['step'],
                                       G['n_steps'])
    return task, dict(per_step=misses.astype(float).tolist(), records=cached_records, lru_pages=lru_pages,
                      rc_entries=entries, rc_pages=rc_pages, rc_hits=int(hit.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='per-query sets of the split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--schedule', required=True, help='arrival metadata of the split (make_schedule.py)')
    ap.add_argument('--train-unions', required=True, help='training group unions; {draw} is replaced by the draw')
    ap.add_argument('--layout', action='append', required=True, metavar='NAME=FILE', help='FILE may contain {draw}')
    ap.add_argument('--draws', default='0,1,2')
    ap.add_argument('--pages', type=int, required=True)
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--budgets', required=True, help='declared budgets in pages, comma separated')
    ap.add_argument('--shares', default=','.join(str(s) for s in cache.SHARE_GRID))
    ap.add_argument('--at', default=None,
                    help='select_deployment.py output, or a JSON map "layout|budget" -> share: replay each pair at '
                         'that share only')
    ap.add_argument('--rc-share', type=float, required=True,
                    help='phi, the share of the budget held by the result cache')
    ap.add_argument('--entry-bytes', type=int, default=72, help='bytes per result-cache entry')
    ap.add_argument('--page-bytes', type=int, default=8192)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    q = np.load(a.sets); s = np.load(a.schedule)
    G.update(e_ptr=q['e_ptr'].astype(np.int64), e_nodes=q['e_nodes'].astype(np.int32), step=s['step'].astype(np.int64),
             n_steps=int(s['n_steps']), pages=a.pages, capacity=a.capacity, unions=a.train_unions, rank={},
             layouts=dict(item.split('=', 1) for item in a.layout),
             order=cache.lockstep_order(s['step'], s['turn'], s['rollout'], s['position']))
    draws = [int(x) for x in a.draws.split(',')]
    budgets = [int(x) for x in a.budgets.split(',')]
    shares = [float(x) for x in a.shares.split(',')]
    G['n'] = len(np.load(next(iter(G['layouts'].values())).format(draw=draws[0]), mmap_mode='r'))
    G['rc'] = {}
    for b in budgets:
        entries = int(a.rc_share * a.page_bytes * b // a.entry_bytes)
        G['rc'][b] = (entries, -(-entries * a.entry_bytes // a.page_bytes),
                      result_cache_hits(G['order'], s['string'], entries))
    at = json.load(open(a.at)) if a.at else None
    at = at.get('shares', at) if at else None   # a select_deployment.py output or a plain map
    tasks = sorted({(name, k, b, sh) for name in G['layouts'] for b in budgets
                    for sh in ([float(at[f'{name}|{b}'])] if at else shares) for k in draws})
    t0 = time.time()
    result = dict(sets=a.sets, pages=a.pages, capacity=a.capacity, budgets=budgets, shares=shares, draws=draws,
                  order='lockstep', n_steps=G['n_steps'], rc_share=a.rc_share, rows={})
    mp.set_start_method('fork', force=True)
    with mp.Pool(a.workers) as pool:
        for task, row in pool.imap_unordered(replay_one, tasks):
            result['rows']['|'.join(str(x) for x in task)] = row
    result['seconds'] = time.time() - t0
    json.dump(result, open(a.out, 'w'))
    print(f'{len(tasks)} replays with result-cache share {a.rc_share} in {result["seconds"]:.0f} s -> {a.out}')


if __name__ == '__main__':
    main()
