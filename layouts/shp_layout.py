"""SHP-G baseline: SHP's rounds of paired swaps on the P05 objective, with the candidate pages of Algorithm 1."""
# [Paper] Section VII-A, Compared methods, "SHP-G: SHP's paired moves on the P05 objective, the move rule only";
# Table III, row SHP-G; Table VI, row SHP-G; Section VII-C (gain predicted on a round's snapshot against the realized
# decrease).
# Each round works on a snapshot of the layout: every covered record proposes the candidate page with the largest
# single-move gain if that gain is positive; for every page pair {a, b} the a->b and b->a proposers are sorted by gain
# and the i-th of each list swap. Only swaps are applied, so a round can raise the objective, as in SHP.
import argparse
import json
import time

import numpy as np
import numba as nb

from hyperpage.formats import read_index_meta, parse_graph
from hyperpage.hypergraph import incidence
from hyperpage.objectives import fanout_table, fanout_value
from hyperpage.refine import count_get, count_add, build_count_tables, slots_of_layout
from hyperpage.weights import arc_pairs, pairs_to_csr


@nb.njit(cache=True)
def propose(page, npages, B, e_ptr, e_nodes, v_ptr, v_edges, t_ptr, t_keys, t_vals, frozen, v_pos, max_sets, window,
            adj_ptr, adj_idx, fan_step, max_cand, frozen_on_page, covered):
    """Best single-move target page of every covered record among its candidate pages, on the current count tables.

    Candidate pages are those of Algorithm 1 (Section VI-C). Returns (record, from page, to page, gain) of the
    records whose best move has a positive gain."""
    nt_all = len(covered)
    pv = np.empty(nt_all, np.int32); pa = np.empty(nt_all, np.int32); pb = np.empty(nt_all, np.int32)
    pg = np.empty(nt_all, np.int64)
    score = np.zeros(npages, np.int64); seen_pages = np.empty(npages, np.int32)
    k = 0
    for v in covered:
        if frozen[v]:
            continue
        a = page[v]
        nt = 0
        for t in range(adj_ptr[v], adj_ptr[v + 1]):
            p = page[adj_idx[t]]
            if p == a or frozen_on_page[p] >= B:
                continue
            if score[p] == 0:
                seen_pages[nt] = p; nt += 1
            score[p] += 1
        ne = 0
        for t in range(v_ptr[v], v_ptr[v + 1]):
            ne += 1
            if ne > max_sets:
                break
            e = v_edges[t]
            lo = e_ptr[e] + max(v_pos[t] - window, 0)
            hi = min(e_ptr[e] + v_pos[t] + window + 1, e_ptr[e + 1])
            for j in range(lo, hi):
                p = page[e_nodes[j]]
                if p == a or frozen_on_page[p] >= B:
                    continue
                if score[p] == 0:
                    seen_pages[nt] = p; nt += 1
                score[p] += 1
        if nt == 0:
            continue
        if nt > max_cand:
            av = np.empty(nt, np.int64)
            for i in range(nt):
                av[i] = -score[seen_pages[i]]
            idx = np.argsort(av, kind='mergesort')[:max_cand]
            cands = seen_pages[idx].copy()
        else:
            cands = seen_pages[:nt].copy()
        for i in range(nt):
            score[seen_pages[i]] = 0
        best = np.int64(0); bb = -1
        for b in cands:
            dv = np.int64(0)
            for t in range(v_ptr[v], v_ptr[v + 1]):
                e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
                dv += (fan_step[count_get(t_keys, t_vals, base, cap, a) - 1]
                       - fan_step[count_get(t_keys, t_vals, base, cap, b)])
            if dv > best:
                best = dv; bb = b
        if bb >= 0:
            pv[k] = v; pa[k] = a; pb[k] = bb; pg[k] = best; k += 1
    return pv[:k], pa[:k], pb[:k], pg[:k]


@nb.njit(cache=True)
def pair_proposals(lo, hi, d, pg, order):
    """Pairs (i, j) of opposite proposals between the same two pages.

    order sorts the proposals by (lower page, higher page, direction, gain descending); per page pair the i-th a->b
    proposer is paired with the i-th b->a proposer, min(S_ab, S_ba) pairs, as in SHP's pairing of moves."""
    n = len(order)
    si = np.empty(n, np.int64); sj = np.empty(n, np.int64); k = 0
    s = 0
    while s < n:
        t = s
        while t < n and lo[order[t]] == lo[order[s]] and hi[order[t]] == hi[order[s]]:
            t += 1
        m0 = s
        while m0 < t and d[order[m0]] == 0:
            m0 += 1
        c0 = m0 - s; c1 = t - m0
        for i in range(min(c0, c1)):
            x = order[s + i]; y = order[m0 + i]
            if pg[x] + pg[y] > 0:
                si[k] = x; sj[k] = y; k += 1
        s = t
    return si[:k], sj[:k]


@nb.njit(cache=True)
def apply_swaps(sv, su, page, slots, B, v_ptr, v_edges, t_ptr, t_keys, t_vals, fan_step, in_v):
    """Apply the swaps one after another; returns the summed exact decrease of the objective on the current tables."""
    tot = np.int64(0)
    for q in range(len(sv)):
        v = sv[q]; u = su[q]; a = page[v]; b = page[u]
        for t in range(v_ptr[v], v_ptr[v + 1]):
            in_v[v_edges[t]] = 1
        dv = np.int64(0)
        for t in range(v_ptr[v], v_ptr[v + 1]):
            e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
            dv += (fan_step[count_get(t_keys, t_vals, base, cap, a) - 1]
                   - fan_step[count_get(t_keys, t_vals, base, cap, b)])
        du = np.int64(0)
        for t in range(v_ptr[u], v_ptr[u + 1]):
            e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
            ca = count_get(t_keys, t_vals, base, cap, a); cb = count_get(t_keys, t_vals, base, cap, b)
            # sets that hold both records keep their counts on a and b
            if in_v[e]:
                du -= fan_step[ca - 1] - fan_step[cb]
            else:
                du += fan_step[cb - 1] - fan_step[ca]
        for t in range(v_ptr[v], v_ptr[v + 1]):
            in_v[v_edges[t]] = 0
        tot += dv + du
        for t in range(v_ptr[v], v_ptr[v + 1]):
            e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
            count_add(t_keys, t_vals, base, cap, a, -1); count_add(t_keys, t_vals, base, cap, b, 1)
        for t in range(v_ptr[u], v_ptr[u + 1]):
            e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
            count_add(t_keys, t_vals, base, cap, b, -1); count_add(t_keys, t_vals, base, cap, a, 1)
        sa = -1; sb = -1
        for s in range(B):
            if slots[a * B + s] == v:
                sa = s
            if slots[b * B + s] == u:
                sb = s
        slots[a * B + sa] = u; slots[b * B + sb] = v
        page[v] = b; page[u] = a
    return tot


def shp_round(page, slots, B, npages, e_ptr, e_nodes, v_ptr, v_edges, t_ptr, t_keys, t_vals, frozen, v_pos, adjacency,
              fan_step, frozen_on_page, covered, in_v, max_cand, max_sets, window):
    """One SHP round: proposals on the snapshot, pairing, then the swaps applied in sequence."""
    pv, pa, pb, pg = propose(page, npages, B, e_ptr, e_nodes, v_ptr, v_edges, t_ptr, t_keys, t_vals, frozen, v_pos,
                             max_sets, window, adjacency[0], adjacency[1], fan_step, max_cand, frozen_on_page, covered)
    lo = np.minimum(pa, pb); hi = np.maximum(pa, pb); d = (pa > pb).astype(np.int8)
    order = np.lexsort((-pg, d, hi, lo)).astype(np.int64)
    si, sj = pair_proposals(lo, hi, d, pg, order)
    # gain that the snapshot predicts for the paired swaps, against the decrease they realize when applied in sequence
    snapshot_gain = int(pg[si].sum() + pg[sj].sum())
    sv = pv[si].copy(); su = pv[sj].copy()
    decrease = int(apply_swaps(sv, su, page, slots, B, v_ptr, v_edges, t_ptr, t_keys, t_vals, fan_step, in_v))
    pairs_both = int(len(np.unique(lo[si] * np.int64(npages) + hi[si]))) if len(si) else 0
    return dict(proposals=int(len(pv)), swaps=int(len(sv)), page_pairs_swapping=pairs_both, snapshot_gain=snapshot_gain,
                decrease=decrease)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index-prefix', required=True,
                    help='prefix of the index files (<prefix>_disk.index, <prefix>_mem.index)')
    ap.add_argument('--sets', required=True, help='training group unions of one draw (.npz with e_ptr, e_nodes)')
    ap.add_argument('--start', required=True, help="start layout (.npy); the paper uses MARGO's")
    ap.add_argument('--capacity', type=int, default=12,
                    help='records per page, B (at most 12 for the integer P05 table)')
    ap.add_argument('--max-rounds', type=int, default=60, help='stop after this many rounds')
    ap.add_argument('--cpu-budget', type=float, default=None,
                    help='also stop once the rounds have used this many process CPU seconds (no limit if omitted)')
    ap.add_argument('--max-cand', type=int, default=16, help='candidate pages per record')
    ap.add_argument('--max-sets', type=int, default=64, help='sets per record scanned for candidate pages')
    ap.add_argument('--window', type=int, default=32,
                    help='positions around the record inside a set scanned for candidate pages')
    ap.add_argument('--out', required=True,
                    help='output layout (.npy); run details and the per-round history go to the same name with .json')
    a = ap.parse_args()

    B = a.capacity
    assert B <= 12, 'the integer fanout table covers page capacities up to 12'
    n = read_index_meta(a.index_prefix + '_disk.index')['npts']
    z = np.load(a.sets)
    e_ptr, e_nodes, v_ptr, v_edges, v_pos = incidence(z['e_ptr'], z['e_nodes'], n)
    m = len(e_ptr) - 1
    degree = np.diff(v_ptr)
    frozen = np.zeros(n, bool)
    g_ptr, g_nbr = parse_graph(np.fromfile(a.index_prefix + '_mem.index', dtype=np.uint32), n)
    # [Paper] Section VI-C: candidate pages include the pages of graph neighbours inside training sets
    adjacency = pairs_to_csr(*arc_pairs(e_ptr, e_nodes, g_ptr, g_nbr, n), n)[:2]
    del g_ptr, g_nbr
    fan, fan_step = fanout_table()
    npages = -(-n // B)
    page = np.load(a.start).astype(np.int32)
    assert len(page) == n, 'the start layout does not match the index'
    page0 = page.copy()
    value0 = int(fanout_value(e_ptr, e_nodes, page, npages, fan))
    slots = slots_of_layout(page, npages, B)
    t_ptr, t_keys, t_vals = build_count_tables(e_ptr, e_nodes, page)
    # records outside the training sets never propose, so they never move
    covered = np.nonzero(degree > 0)[0].astype(np.int32)
    frozen_on_page = np.zeros(npages, np.int64)
    in_v = np.zeros(max(m, 1), np.int8)

    # compile the round on an empty proposal set so that compilation is not charged to the CPU budget
    shp_round(page.copy(), slots.copy(), B, npages, e_ptr, e_nodes, v_ptr, v_edges, t_ptr.copy(), t_keys.copy(),
              t_vals.copy(), frozen, v_pos, adjacency, fan_step, frozen_on_page, covered[:0], in_v, a.max_cand,
              a.max_sets, a.window)
    history = []; value = value0; best = value0; best_round = 0; cum_cpu = 0.0; cum_wall = 0.0
    for r in range(1, a.max_rounds + 1):
        c0, w0 = time.process_time(), time.time()
        row = shp_round(page, slots, B, npages, e_ptr, e_nodes, v_ptr, v_edges, t_ptr, t_keys, t_vals, frozen, v_pos,
                        adjacency, fan_step, frozen_on_page, covered, in_v, a.max_cand, a.max_sets, a.window)
        dc, dw = time.process_time() - c0, time.time() - w0; cum_cpu += dc; cum_wall += dw
        value -= row['decrease']
        # the maintained value must equal the objective recomputed from the layout (outside the budget)
        check = int(fanout_value(e_ptr, e_nodes, page, npages, fan))
        assert check == value, (r, check, value)
        row.update(round=r, value=value, cpu_seconds=dc, wall_seconds=dw, cum_cpu_seconds=cum_cpu,
                   cum_wall_seconds=cum_wall)
        history.append(row); print(json.dumps(row), flush=True)
        if value < best:
            best, best_round = value, r
        if row['swaps'] == 0 or (a.cpu_budget is not None and cum_cpu >= a.cpu_budget):
            break
    counts = np.bincount(page, minlength=npages)
    assert np.array_equal(counts, np.bincount(page0, minlength=npages)) and counts.max() <= B
    if history[-1]['swaps'] == 0:
        stop = 'no swap'
    elif a.cpu_budget is not None and cum_cpu >= a.cpu_budget:
        stop = 'cpu budget'
    else:
        stop = 'round cap'
    np.save(a.out, page.astype(np.int32))
    info = dict(layout='SHP-G', capacity=B, records=int(n), pages=int(npages), sets=int(m), covered=int(len(covered)),
                value_start=value0, value_end=value, best_value=best, best_round=best_round, rounds=len(history),
                stop=stop, cpu_budget=a.cpu_budget, max_rounds=a.max_rounds, moved_records=int((page != page0).sum()),
                history=history)
    json.dump(info, open(a.out[:-4] + '.json' if a.out.endswith('.npy') else a.out + '.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != 'history'}))


if __name__ == '__main__':
    main()
