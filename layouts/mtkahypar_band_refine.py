"""Mt-KaHyPar baseline as refinement: V-cycles of the connectivity (km1) objective from MARGO's layout, run within
bands of consecutive pages at exact page capacity; cache-blind or on the residual unions of one budget.
Needs the mtkahypar Python package, which is not shipped."""
# [Paper] Section VII-A, Compared methods, "Mt-KaHyPar ... At 21M it refines MARGO's layout with its V-cycles within
# bands of 43 consecutive pages"; Table III, rows Mt-KaHyPar (21M) and "on residual unions" (1M at 25%, all 21M
# budgets); Table VI, row Mt-KaHyPar (21M).
import argparse
import json
import resource
import time
from collections import defaultdict

import numpy as np

from hyperpage.hypergraph import most_frequent_records, residual_sets


def band_pieces(sets, band_of):
    """Pieces of every set inside each band with at least two pins, in band order per set and pin order per piece."""
    out = {}
    for s in sets:
        bs = band_of[s]; o = np.argsort(bs, kind='stable'); bo, so = bs[o], s[o]
        cut = np.flatnonzero(np.diff(bo)) + 1
        for lo, hi in zip(np.concatenate([[0], cut]), np.concatenate([cut, [len(so)]])):
            if hi - lo >= 2:
                out.setdefault(int(bo[lo]), []).append(so[lo:hi])
    return out


def repair(blk, sets_of, k, B):
    """Greedy repair of a band partition to at most B records per block (a no-op when every block fits).

    sets_of[v] lists the local sets of local vertex v. While a block holds more than B records, it moves the
    record and destination block with room whose move adds the least connectivity, sum over the record's sets of
    [no pin of the set in the destination] - [the record is the set's only pin in its block]."""
    cnt = np.bincount(blk, minlength=k); moved = 0
    if cnt.max() <= B:
        return blk, 0
    pins_in = defaultdict(int)
    for v, ss in enumerate(sets_of):
        for s in ss:
            pins_in[(s, blk[v])] += 1
    room = [b for b in range(k) if cnt[b] < B]
    for b in np.nonzero(cnt > B)[0]:
        while cnt[b] > B:
            mem = np.nonzero(blk == b)[0]; best = None
            for v in mem:
                for d in room:
                    if cnt[d] >= B:
                        continue
                    c = sum((pins_in[(s, d)] == 0) - (pins_in[(s, b)] == 1) for s in sets_of[v])
                    if best is None or c < best[0]:
                        best = (c, v, d)
            c, v, d = best
            for s in sets_of[v]:
                pins_in[(s, b)] -= 1; pins_in[(s, d)] += 1
            blk[v] = d; cnt[b] -= 1; cnt[d] += 1; moved += 1
            room = [x for x in room if cnt[x] < B]
    return blk, moved


def connectivity_sum(ptr, nodes, page):
    """sum_E lambda_pi(E) over the non-empty sets."""
    return int(sum(len(np.unique(page[nodes[ptr[i]:ptr[i + 1]]])) for i in range(len(ptr) - 1) if ptr[i + 1] > ptr[i]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='training group unions of one draw (.npz with e_ptr, e_nodes)')
    ap.add_argument('--start', required=True, help="MARGO's layout (.npy), the initial partition")
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--seed', type=int, default=0,
                    help="Mt-KaHyPar's seed; the paper uses the index of the training draw")
    ap.add_argument('--cached-pages', type=int, default=0,
                    help='refine on the residual unions: drop the capacity * cached-pages most frequent training '
                         'records, cached-pages = floor(rho * D)')
    ap.add_argument('--band-pages', type=int, default=43, help='consecutive start pages per band')
    ap.add_argument('--vcycles', type=int, default=3, help='V-cycles of improve_partition per band')
    ap.add_argument('--preset', default='QUALITY', help='Mt-KaHyPar preset')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--eps', type=float, default=0.03, help='imbalance passed with the exact block weights')
    ap.add_argument('--out', required=True, help='output layout (.npy); run details go to the same name with .json')
    a = ap.parse_args()

    import mtkahypar
    B = a.capacity
    ctx_factory = mtkahypar.initialize(a.threads); mtkahypar.set_seed(int(a.seed))
    t0 = time.time()
    start = np.load(a.start).astype(np.int64); n = len(start); npages = int(start.max()) + 1
    assert np.bincount(start, minlength=npages).max() <= B
    z = np.load(a.sets); e_ptr = z['e_ptr'].astype(np.int64); e_nodes = z['e_nodes'].astype(np.int64)
    hot = np.zeros(n, bool)
    if a.cached_pages > 0:
        # [Paper] Definition 7: the record cache holds the B * floor(rho * D) most frequent training records
        hot = most_frequent_records(e_ptr, e_nodes, n, B * a.cached_pages)
    r_ptr, r_nodes = residual_sets(e_ptr, e_nodes, hot)
    sets = [r_nodes[r_ptr[i]:r_ptr[i + 1]] for i in range(len(r_ptr) - 1) if r_ptr[i + 1] - r_ptr[i] >= 2]

    def refine_band(n_verts, band_sets, init_blk, k):
        # every record is a vertex of weight 1 and every page a block of weight at most B
        ctx = ctx_factory.context_from_preset(getattr(mtkahypar.PresetType, a.preset)); ctx.logging = False
        ctx.set_partitioning_parameters(int(k), a.eps, mtkahypar.Objective.KM1)
        ctx.set_individual_target_block_weights([B] * int(k))
        hg = ctx_factory.create_hypergraph(ctx, n_verts, len(band_sets), band_sets, [1] * n_verts, [1] * len(band_sets))
        ph = hg.create_partitioned_hypergraph(ctx, int(k), init_blk.tolist())
        before = int(ph.km1())
        if band_sets:
            ph.improve_partition(ctx, a.vcycles)
        return np.array(ph.get_partition(), np.int64), before, int(ph.km1())

    page = start.copy(); stats = dict(km1_before=0, km1_after=0, moved_repair=0, bands=0)
    order = np.argsort(start, kind='stable')
    pptr = np.concatenate([[0], np.cumsum(np.bincount(start, minlength=npages))])
    band_of = start // a.band_pages; n_bands = int(band_of.max()) + 1
    pieces = band_pieces(sets, band_of)
    for bnd in range(n_bands):
        # records move only among the pages of their band; the sets are restricted to the band
        p_lo, p_hi = bnd * a.band_pages, min(npages, (bnd + 1) * a.band_pages)
        verts = order[pptr[p_lo]:pptr[p_hi]]; k = p_hi - p_lo
        loc = {int(v): j for j, v in enumerate(verts)}
        band_sets = [[loc[int(v)] for v in pins] for pins in pieces.get(bnd, [])]
        blk, before, after = refine_band(len(verts), band_sets, start[verts] - p_lo, k)
        sets_of = [[] for _ in range(len(verts))]
        for i, s in enumerate(band_sets):
            for v in s:
                sets_of[v].append(i)
        # Mt-KaHyPar can leave a block above B; the greedy repair restores exact capacity within the band
        blk, moved = repair(blk, sets_of, k, B)
        page[verts] = blk + p_lo
        stats['km1_before'] += before; stats['km1_after'] += after; stats['moved_repair'] += moved
        stats['bands'] += 1
    page = page.astype(np.int32)
    assert np.bincount(page, minlength=npages).max() <= B and len(page) == n

    ru = resource.getrusage(resource.RUSAGE_SELF)
    info = dict(layout='Mt-KaHyPar band refinement', cached_pages=a.cached_pages, preset=a.preset, vcycles=a.vcycles,
                band_pages=a.band_pages, threads=a.threads, eps=a.eps, seed=a.seed, capacity=B, pages=npages,
                stats=stats, seconds=time.time() - t0, cpu_seconds=ru.ru_utime + ru.ru_stime,
                moved_records=int((page != start).sum()),
                residual_connectivity_sum=connectivity_sum(r_ptr, r_nodes, page),
                connectivity_sum=connectivity_sum(e_ptr, e_nodes, page),
                start_residual_connectivity_sum=connectivity_sum(r_ptr, r_nodes, start.astype(np.int32)))
    np.save(a.out, page)
    json.dump(info, open(a.out[:-4] + '.json' if a.out.endswith('.npy') else a.out + '.json', 'w'), indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
