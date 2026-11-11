"""Stage the inputs of hybrid-cache engine runs for one split: sub-queries in turn lockstep, their waves, the hot-record
list of every arm and, optionally, the replayed per-step reads that a one-thread run must reproduce."""
# [Paper] Section VI-E and Definition 7. Waves are maximal runs of equal (step, turn) in turn lockstep (Section VII-A,
# Metrics); hot records are the B * floor(rho * D) records most frequent in training draw 0; the replay reference uses
# the replay's own hybrid-cache kernel (hyperpage.cache) in the same order.
import argparse
import json
import os

import numpy as np

from hyperpage import cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='per-query sets of the split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--schedule', required=True,
                    help='arrival metadata of the split from replay/make_schedule.py (step, turn, rollout, position, '
                         'string, n_steps)')
    ap.add_argument('--key', required=True, help='split key, e.g. val or test (names the output files)')
    ap.add_argument('--queries', required=True, help='int8 query vectors by string id (.npy)')
    ap.add_argument('--train-unions', required=True,
                    help='training group unions of draw 0 (.npz), which rank hot records')
    ap.add_argument('--records', type=int, required=True, help='records (nodes) of the index')
    ap.add_argument('--pages', type=int, required=True, help='pages of the index')
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--budget', type=int, required=True, help='budget D in pages')
    ap.add_argument('--arm', nargs=3, action='append', default=[], metavar=('NAME', 'LAYOUT', 'SHARE'),
                    help='arm name, layout (.npy) and hot-record share rho; repeat for every arm')
    ap.add_argument('--no-reference', action='store_true', help='skip the replayed per-step reads')
    ap.add_argument('--out', required=True, help='output directory')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Q = np.load(a.sets); e_ptr, e_nodes = Q['e_ptr'].astype(np.int64), Q['e_nodes'].astype(np.int32)
    s = np.load(a.schedule)
    step = s['step'].astype(np.int64); ns = int(s['n_steps']); nc = len(step)
    strid = s['string'].astype(np.int64)   # distinct-string id of every sub-query, in split order
    assert nc == len(e_ptr) - 1 == len(strid), (nc, len(e_ptr) - 1, len(strid))

    # turn lockstep and its waves: one wave per (step, turn)
    lock = cache.lockstep_order(s['step'], s['turn'], s['rollout'], s['position'])
    ls, lt = step[lock], s['turn'][lock]
    brk = np.flatnonzero((np.diff(ls) != 0) | (np.diff(lt) != 0)) + 1
    wptr = np.concatenate([[0], brk, [nc]]).astype(np.uint64); wstep = ls[wptr[:-1].astype(np.int64)].astype(np.uint32)
    wptr.tofile(f'{a.out}/{a.key}_wptr.u64'); wstep.tofile(f'{a.out}/{a.key}_wstep.u32')
    np.save(f'{a.out}/{a.key}_lock.npy', lock)
    Q8 = np.load(a.queries, mmap_mode='r'); X = np.ascontiguousarray(Q8[strid[lock]])
    with open(f'{a.out}/{a.key}_q.bin', 'wb') as f:
        np.array([X.shape[0], X.shape[1]], np.int32).tofile(f); X.tofile(f)

    z = np.load(a.train_unions)
    freq = cache.record_frequency(z['e_ptr'].astype(np.int64), z['e_nodes'].astype(np.int32), a.records)
    ranking = cache.rank_records(freq)
    info = dict(split=a.key, calls=int(nc), waves=int(len(wptr) - 1), steps=ns, budget_pages=a.budget, arms={})
    for name, lay, share in a.arm:
        hot_pages, cached_records, lru_pages = cache.split_budget(a.budget, float(share), a.capacity)
        hot = ranking[:cached_records].astype(np.uint32); hot.tofile(f'{a.out}/hot_{name}.u32')
        row = dict(layout=os.path.basename(lay), share=float(share), hot_pages=hot_pages, hot_records=cached_records,
                   lru_pages=lru_pages)
        if not a.no_reference:
            pg = np.load(lay).astype(np.int32); assert int(pg.max()) + 1 <= a.pages
            mask = cache.cached_mask(pg, a.pages, ranking, freq, hot_pages, cached_records)
            miss = cache.hybrid_cache_misses(lock, e_ptr, e_nodes, pg, a.pages, lru_pages, mask, step, ns)
            np.save(f'{a.out}/ref_{a.key}_{name}.npy', miss); row['reads'] = int(miss.sum())
        info['arms'][name] = row
    json.dump(info, open(f'{a.out}/{a.key}_info.json', 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
