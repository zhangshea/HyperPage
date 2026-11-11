"""Refinement of a page layout by exact moves and swaps under a hard page capacity (Algorithm 1)."""
import numpy as np
import numba as nb


@nb.njit(inline='always')
def _slot_hash(p, mask):
    return (np.int64(p) * np.int64(2654435761)) & mask


@nb.njit(inline='always')
def count_get(t_keys, t_vals, base, cap, p):
    """Page count c_{E,p} of one set (0 when the page is absent)."""
    mask = cap - 1
    i = _slot_hash(p, mask)
    while True:
        key = t_keys[base + i]
        if key == p:
            return t_vals[base + i]
        if key == -1:
            return 0
        i = (i + 1) & mask


@nb.njit(inline='always')
def count_add(t_keys, t_vals, base, cap, p, delta):
    """Add delta to the count of page p in one set; a count that reaches 0 is deleted by backward shift."""
    mask = cap - 1
    i = _slot_hash(p, mask)
    while True:
        key = t_keys[base + i]
        if key == p:
            t_vals[base + i] += delta
            if t_vals[base + i] == 0:
                j = i
                while True:
                    j = (j + 1) & mask
                    key_j = t_keys[base + j]
                    if key_j == -1:
                        break
                    h = _slot_hash(key_j, mask)
                    if i <= j:
                        if i < h and h <= j:
                            continue
                    else:
                        if i < h or h <= j:
                            continue
                    t_keys[base + i] = key_j
                    t_vals[base + i] = t_vals[base + j]
                    i = j
                t_keys[base + i] = -1
                t_vals[base + i] = 0
            return
        if key == -1:
            t_keys[base + i] = p
            t_vals[base + i] = delta
            return
        i = (i + 1) & mask


@nb.njit(cache=True)
def build_count_tables(e_ptr, e_nodes, page):
    """Page-count tables: one open-addressing hash table per set that maps a page p to c_{E,p}.

    Returns (t_ptr, t_keys, t_vals); the table of set e occupies t_keys[t_ptr[e]:t_ptr[e + 1]], with -1 for an empty cell."""
    m = len(e_ptr) - 1
    t_ptr = np.zeros(m + 1, np.int64)
    for e in range(m):
        sz = e_ptr[e + 1] - e_ptr[e]
        cap = 4
        while cap < 2 * sz:
            cap *= 2
        t_ptr[e + 1] = t_ptr[e] + cap
    t_keys = np.full(t_ptr[m], -1, np.int32)
    t_vals = np.zeros(t_ptr[m], np.int32)
    for e in range(m):
        base = t_ptr[e]; cap = t_ptr[e + 1] - base
        for j in range(e_ptr[e], e_ptr[e + 1]):
            count_add(t_keys, t_vals, base, cap, page[e_nodes[j]], 1)
    return t_ptr, t_keys, t_vals


def slots_of_layout(page, npages, B):
    """Slot table of a layout: slots[p * B + s] is the record in slot s of page p, or -1 for a free slot."""
    n = len(page)
    slots = np.full(npages * B, -1, np.int32)
    order = np.argsort(page, kind='stable')
    pg = page[order]
    start = np.searchsorted(pg, np.arange(npages))
    rank = np.arange(n) - start[pg]
    assert rank.max() < B
    slots[pg.astype(np.int64) * B + rank] = order
    return slots


@nb.njit(cache=True)
def refine(mode, page, slots, B, npages, e_ptr, e_nodes, v_ptr, v_edges, t_ptr, t_keys, t_vals,
           w_ptr, w_nbr, w_val, order, max_cand, max_passes, tol, value0, frozen, v_pos, max_sets, window,
           adj_ptr, adj_idx, fan_step):
    """Strict-descent refinement of a layout by exact moves and swaps (in place).

    mode selects the objective (objectives.SET, CLQ, PAIRWISE or P05). page[v] is the page of record v and
    slots[p * B + s] the record in slot s of page p, or -1 for a free slot. (e_ptr, e_nodes) are the sets,
    (v_ptr, v_edges, v_pos) their incidence lists with the position of each record inside its set, and
    (t_ptr, t_keys, t_vals) the page-count tables of build_count_tables. (w_ptr, w_nbr, w_val) is the weighted graph of
    the PAIRWISE objective. order lists the records that may initiate an operation, in the order they are visited.
    A frozen record never moves and is never a swap partner. Candidate pages of v are the pages of its set co-members
    within window positions in up to max_sets of its sets, plus the pages of its neighbours in (adj_ptr, adj_idx); the
    max_cand pages with the most such records are tried. value0 is the objective value of the start layout.

    Returns (value, passes, n_applied, pass_gain): the final objective value, the number of passes run, the number of
    applied operations and the decrease of every pass."""
    n = len(page)
    frozen_on_page = np.zeros(npages, np.int64)
    for x in range(n):
        if frozen[x]:
            frozen_on_page[page[x]] += 1
    m = len(e_ptr) - 1
    score = np.zeros(npages, np.int64)
    seen_pages = np.empty(npages, np.int32)
    in_v = np.zeros(max(m, 1), np.int8)
    w_of_v = np.zeros(n, np.int64)
    value = value0
    passes = 0
    n_applied = 0
    pass_gain = np.zeros(max_passes, np.int64)
    # [Paper] Algorithm 1, lines 2 to 9
    for it in range(max_passes):
        gain = 0
        for v in order:
            if frozen[v]:
                continue
            # [Paper] Algorithm 1, line 4: candidate pages of v (Section VI-C)
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
                cand_pages = seen_pages[idx].copy()
            else:
                cand_pages = seen_pages[:nt].copy()
            for i in range(nt):
                score[seen_pages[i]] = 0
            if mode == 2:
                for t in range(w_ptr[v], w_ptr[v + 1]):
                    w_of_v[w_nbr[t]] = w_val[t]
                w_va = 0
                for t in range(w_ptr[v], w_ptr[v + 1]):
                    if page[w_nbr[t]] == a:
                        w_va += w_val[t]
            else:
                for t in range(v_ptr[v], v_ptr[v + 1]):
                    in_v[v_edges[t]] = 1
            # [Paper] Algorithm 1, line 5: exact change of every move into a free slot and every swap
            best = np.int64(0); best_u = -2; best_page = -1; best_slot = -1
            for b in cand_pages:
                gain_v = np.int64(0)
                if mode == 0:
                    for t in range(v_ptr[v], v_ptr[v + 1]):
                        e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
                        ca = count_get(t_keys, t_vals, base, cap, a); cb = count_get(t_keys, t_vals, base, cap, b)
                        # [Paper] Eq. (4), negated (a gain is a decrease): sets that lose page a minus sets that gain page b
                        gain_v += (1 if ca == 1 else 0) - (1 if cb == 0 else 0)
                elif mode == 1:
                    for t in range(v_ptr[v], v_ptr[v + 1]):
                        e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
                        gain_v += count_get(t_keys, t_vals, base, cap, b) - count_get(t_keys, t_vals, base, cap, a) + 1
                elif mode == 3:
                    for t in range(v_ptr[v], v_ptr[v + 1]):
                        e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
                        gain_v += fan_step[count_get(t_keys, t_vals, base, cap, a) - 1] - fan_step[count_get(t_keys, t_vals, base, cap, b)]
                else:
                    w_vb = 0
                    for t in range(w_ptr[v], w_ptr[v + 1]):
                        if page[w_nbr[t]] == b:
                            w_vb += w_val[t]
                    gain_v = w_vb - w_va
                for s in range(B):
                    u = slots[b * B + s]
                    if u >= 0 and frozen[u]:
                        continue
                    if u < 0:
                        gain_uv = gain_v
                    elif mode == 0:
                        gain_u = np.int64(0)
                        for t in range(v_ptr[u], v_ptr[u + 1]):
                            e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
                            ca = count_get(t_keys, t_vals, base, cap, a); cb = count_get(t_keys, t_vals, base, cap, b)
                            gain_u += (1 if cb == 1 else 0) - (1 if ca == 0 else 0)
                            if in_v[e]:
                                # [Paper] Eq. (5): correction for the sets that contain both records
                                gain_u -= (1 if ca == 1 else 0) + (1 if cb == 1 else 0)
                        gain_uv = gain_v + gain_u
                    elif mode == 1:
                        gain_u = np.int64(0)
                        for t in range(v_ptr[u], v_ptr[u + 1]):
                            e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
                            gain_u += count_get(t_keys, t_vals, base, cap, a) - count_get(t_keys, t_vals, base, cap, b) + 1
                            if in_v[e]:
                                gain_u -= 2
                        gain_uv = gain_v + gain_u
                    elif mode == 3:
                        gain_u = np.int64(0)
                        for t in range(v_ptr[u], v_ptr[u + 1]):
                            e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
                            ca = count_get(t_keys, t_vals, base, cap, a); cb = count_get(t_keys, t_vals, base, cap, b)
                            if in_v[e]:
                                gain_u -= fan_step[ca - 1] - fan_step[cb]
                            else:
                                gain_u += fan_step[cb - 1] - fan_step[ca]
                        gain_uv = gain_v + gain_u
                    else:
                        w_ua = 0; w_ub = 0
                        for t in range(w_ptr[u], w_ptr[u + 1]):
                            px = page[w_nbr[t]]
                            if px == a:
                                w_ua += w_val[t]
                            elif px == b:
                                w_ub += w_val[t]
                        w_uv = w_of_v[u]
                        gain_uv = (gain_v - w_uv) + (w_ua - w_uv - w_ub)
                    if gain_uv > best:
                        best = gain_uv; best_u = u; best_page = b; best_slot = s
            if mode == 2:
                for t in range(w_ptr[v], w_ptr[v + 1]):
                    w_of_v[w_nbr[t]] = 0
            else:
                for t in range(v_ptr[v], v_ptr[v + 1]):
                    in_v[v_edges[t]] = 0
            # [Paper] Algorithm 1, line 6: apply only a strict decrease
            if best > 0:
                b = best_page; u = best_u
                sv = -1
                for s in range(B):
                    if slots[a * B + s] == v:
                        sv = s
                        break
                if mode != 2:
                    for t in range(v_ptr[v], v_ptr[v + 1]):
                        e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
                        count_add(t_keys, t_vals, base, cap, a, -1)
                        count_add(t_keys, t_vals, base, cap, b, 1)
                    if u >= 0:
                        for t in range(v_ptr[u], v_ptr[u + 1]):
                            e = v_edges[t]; base = t_ptr[e]; cap = t_ptr[e + 1] - base
                            count_add(t_keys, t_vals, base, cap, b, -1)
                            count_add(t_keys, t_vals, base, cap, a, 1)
                page[v] = b
                slots[b * B + best_slot] = v
                if u >= 0:
                    page[u] = a
                    slots[a * B + sv] = u
                else:
                    slots[a * B + sv] = -1
                value -= best
                gain += best
                n_applied += 1
        pass_gain[it] = gain
        passes += 1
        # [Paper] Algorithm 1, line 8: stop when a pass gains at most tol times the objective value
        # (for SET the value is sum_E lambda(E), that is F_conn plus the number of sets)
        if gain <= tol * abs(value):
            break
    return value, passes, n_applied, pass_gain
