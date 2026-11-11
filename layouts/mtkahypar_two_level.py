"""Mt-KaHyPar baseline in two levels: connectivity (km1) partition of the training unions into leaves, then of every
leaf into blocks of at most B records, which become pages; cache-blind or on the residual unions of one budget.
Needs the mtkahypar Python package, which is not shipped."""
# [Paper] Section VII-A, Compared methods, "Mt-KaHyPar ... On group unions it partitions in two levels as in Bandana at
# 1M"; Table III, rows Mt-KaHyPar and "on residual unions" (1M, 1% and 5%); Table VI, row Mt-KaHyPar (1M).
import argparse
import json
import resource
import time

import numpy as np

from hyperpage.hypergraph import most_frequent_records, residual_sets


def km1_partition(mtk, ctx_factory, preset, nv, sets, k, eps, cap=None):
    """Mt-KaHyPar km1 partition of nv unit-weight vertices into k blocks; cap bounds every block weight."""
    ctx = ctx_factory.context_from_preset(getattr(mtk.PresetType, preset)); ctx.logging = False
    ctx.set_partitioning_parameters(int(k), eps, mtk.Objective.KM1)
    if cap is not None:
        ctx.set_individual_target_block_weights([cap] * int(k))
    hg = ctx_factory.create_hypergraph(ctx, nv, len(sets), sets, [1] * nv, [1] * len(sets))
    ph = hg.partition(ctx)
    return np.array(ph.get_partition(), np.int64), int(ph.km1())


def connectivity_sum(ptr, nodes, page):
    """sum_E lambda_pi(E) over the non-empty sets."""
    return int(sum(len(np.unique(page[nodes[ptr[i]:ptr[i + 1]]])) for i in range(len(ptr) - 1) if ptr[i + 1] > ptr[i]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='training group unions of one draw (.npz with e_ptr, e_nodes)')
    ap.add_argument('--start', required=True,
                    help="MARGO's layout (.npy); its page order places the records outside the partition")
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--seed', type=int, default=0,
                    help="Mt-KaHyPar's seed; the paper uses the index of the training draw")
    ap.add_argument('--cached-pages', type=int, default=0,
                    help='partition the residual unions: drop the capacity * cached-pages most frequent training '
                         'records, cached-pages = floor(rho * D)')
    ap.add_argument('--preset', default='QUALITY', help='Mt-KaHyPar preset of both levels')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--eps', type=float, default=0.03, help='imbalance of both levels')
    ap.add_argument('--leaf-records', type=int, default=512, help='records per leaf of the first level')
    ap.add_argument('--out', required=True, help='output layout (.npy); run details go to the same name with .json')
    a = ap.parse_args()

    import mtkahypar
    B = a.capacity
    ctx_factory = mtkahypar.initialize(a.threads); mtkahypar.set_seed(int(a.seed))
    t0 = time.time()
    start = np.load(a.start)
    n = len(start)
    z = np.load(a.sets); e_ptr = z['e_ptr'].astype(np.int64); e_nodes = z['e_nodes'].astype(np.int64)
    hot = np.zeros(n, bool)
    if a.cached_pages > 0:
        # [Paper] Definition 7: the record cache holds the B * floor(rho * D) most frequent training records
        hot = most_frequent_records(e_ptr, e_nodes, n, B * a.cached_pages)
    r_ptr, r_nodes = residual_sets(e_ptr, e_nodes, hot)
    # vertices: the records left in some residual set; sets with at least two of them
    sub = np.nonzero(np.bincount(r_nodes, minlength=n) > 0)[0]; ns = len(sub)
    loc = np.full(n, -1, np.int64); loc[sub] = np.arange(ns)
    sets = [loc[r_nodes[r_ptr[i]:r_ptr[i + 1]]].tolist() for i in range(len(r_ptr) - 1) if r_ptr[i + 1] - r_ptr[i] >= 2]

    # first level: leaves of about leaf-records records, no capacity per block
    K0 = -(-ns // a.leaf_records)
    part, coarse_km1 = km1_partition(mtkahypar, ctx_factory, a.preset, ns, sets, K0, a.eps)
    coarse_seconds = time.time() - t0
    o = np.argsort(part, kind='stable'); ptr = np.concatenate([[0], np.cumsum(np.bincount(part, minlength=K0))])
    set_of_pin = np.concatenate([np.full(len(e), i) for i, e in enumerate(sets)]); pins = np.concatenate(sets)
    oo = np.argsort(pins, kind='stable'); v_sets = set_of_pin[oo]
    v_ptr = np.concatenate([[0], np.cumsum(np.bincount(pins, minlength=ns))])
    lid = np.full(ns, -1, np.int64)
    full, left = [], []
    for li in range(K0):
        lst = o[ptr[li]:ptr[li + 1]]; L = len(lst)
        if L == 0:
            continue
        k = -(-L // B)
        if k == 1:
            (full if L == B else left).append(sub[lst]); continue
        # second level: the sets restricted to the leaf (at least two pins), blocks of at most B records
        lid[lst] = np.arange(L)
        es = np.unique(np.concatenate([v_sets[v_ptr[x]:v_ptr[x + 1]] for x in lst]))
        leaf_sets = []
        for e in es:
            p = lid[np.asarray(sets[e])]; p = p[p >= 0]
            if len(p) >= 2:
                leaf_sets.append(p.tolist())
        lid[lst] = -1
        if not leaf_sets:
            blk = np.arange(L) // B
        else:
            blk = km1_partition(mtkahypar, ctx_factory, a.preset, L, leaf_sets, k, a.eps, cap=B)[0]
        for b in range(k):
            mem = sub[lst[blk == b]]
            if len(mem) > B:
                left.append(mem[B:]); mem = mem[:B]
            (full if len(mem) == B else left).append(mem)
    # full blocks become pages first, then the remainders in block order, then the records outside the partition
    # (cached records and records in no residual set) in the start layout's page order, ties by identifier
    inpart = np.zeros(n, bool); inpart[sub] = True; rest = np.nonzero(~inpart)[0]
    rest = rest[np.lexsort((rest, start[rest]))]
    left = np.concatenate(left) if left else np.zeros(0, np.int64)
    full = np.concatenate(full) if full else np.zeros(0, np.int64)
    seq = np.concatenate([full, left, rest]).astype(np.int64)
    assert len(seq) == n and len(np.unique(seq)) == n
    page = np.empty(n, np.int32); page[seq] = (np.arange(n) // B).astype(np.int32); npages = -(-n // B)
    assert page.max() == npages - 1 and np.bincount(page, minlength=npages).max() <= B

    ru = resource.getrusage(resource.RUSAGE_SELF)
    info = dict(layout='Mt-KaHyPar two levels', cached_pages=a.cached_pages, preset=a.preset, threads=a.threads,
                eps=a.eps,
                leaf_records=a.leaf_records, seed=a.seed, capacity=B, pages=npages, partitioned=int(ns),
                full_pages=int(len(full) // B), leftover=int(len(left)), leaves=int(K0), coarse_km1=coarse_km1,
                coarse_seconds=coarse_seconds, seconds=time.time() - t0, cpu_seconds=ru.ru_utime + ru.ru_stime,
                residual_connectivity_sum=connectivity_sum(r_ptr, r_nodes, page),
                connectivity_sum=connectivity_sum(e_ptr, e_nodes, page),
                start_residual_connectivity_sum=connectivity_sum(r_ptr, r_nodes, start.astype(np.int32)))
    np.save(a.out, page)
    json.dump(info, open(a.out[:-4] + '.json' if a.out.endswith('.npy') else a.out + '.json', 'w'), indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
