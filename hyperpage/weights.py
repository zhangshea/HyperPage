"""Pairwise weights for layout objectives: arc-restricted co-membership (ARC), logged arc heat (HEAT) and in-page arcs (STARLING)."""
import numpy as np
import numba as nb


def pairs_to_csr(u, v, w, n):
    """Symmetric CSR (ptr, nbr, weight) of unique undirected pairs u < v with integer weights."""
    uu = np.concatenate([u, v]); vv = np.concatenate([v, u]); ww = np.concatenate([w, w])
    o = np.argsort(uu, kind='stable')
    uu, vv, ww = uu[o], vv[o], ww[o]
    cnt = np.bincount(uu, minlength=n)
    ptr = np.zeros(n + 1, np.int64); ptr[1:] = np.cumsum(cnt)
    return ptr, vv.astype(np.int32), ww.astype(np.int64)


@nb.njit(cache=True)
def arc_pair_keys(e_ptr, e_nodes, g_ptr, g_nbr, n):
    """Keys u * n + v (u < v) of the pairs of set members joined by a graph arc, once per set.

    An arc in either direction counts, and a pair joined in both directions is counted once per set."""
    m = len(e_ptr) - 1
    mark = np.zeros(n, np.int32) - 1
    cnt = 0
    for e in range(m):
        for j in range(e_ptr[e], e_ptr[e + 1]):
            mark[e_nodes[j]] = e
        for j in range(e_ptr[e], e_ptr[e + 1]):
            u = e_nodes[j]
            for t in range(g_ptr[u], g_ptr[u + 1]):
                if mark[g_nbr[t]] == e:
                    cnt += 1
    out = np.empty(cnt, np.int64)
    c = 0
    mark[:] = -1
    for e in range(m):
        for j in range(e_ptr[e], e_ptr[e + 1]):
            mark[e_nodes[j]] = e
        st = c
        for j in range(e_ptr[e], e_ptr[e + 1]):
            u = e_nodes[j]
            for t in range(g_ptr[u], g_ptr[u + 1]):
                x = g_nbr[t]
                if mark[x] == e:
                    lo = min(u, x); hi = max(u, x)
                    out[c] = lo * n + hi
                    c += 1
        seg = np.sort(out[st:c])
        k = st
        for i in range(len(seg)):
            if i == 0 or seg[i] != seg[i - 1]:
                out[k] = seg[i]; k += 1
        c = k
    return out[:c]


def arc_pairs(e_ptr, e_nodes, g_ptr, g_nbr, n, chunk=8_000_000):
    """Arc-restricted co-membership weights: w_uv = number of sets that contain both ends of a graph arc {u, v}.

    Returns unique pairs (u, v, w) with u < v. The sets are processed in chunks of at most chunk incidences to bound
    memory; the result does not depend on chunk."""
    # [Paper] Definition 9: the ARC objective restricts co-located pairs to the arcs of the index graph
    m = len(e_ptr) - 1
    K = np.zeros(0, np.int64); C = np.zeros(0, np.int64)
    s = 0
    while s < m:
        t = s + 1
        while t < m and e_ptr[t + 1] - e_ptr[s] <= chunk:
            t += 1
        sub_ptr = (e_ptr[s:t + 1] - e_ptr[s]).astype(np.int64)
        sub_nodes = e_nodes[e_ptr[s]:e_ptr[t]]
        keys = arc_pair_keys(sub_ptr, sub_nodes, g_ptr, g_nbr, n)
        k, c = np.unique(keys, return_counts=True); del keys
        if len(K) == 0:
            K, C = k, c.astype(np.int64)
        else:
            allk = np.concatenate([K, k]); allc = np.concatenate([C, c.astype(np.int64)])
            K, inv = np.unique(allk, return_inverse=True); C = np.bincount(inv, weights=allc).astype(np.int64)
            del allk, allc, inv
        s = t
    return (K // n).astype(np.int64), (K % n).astype(np.int64), C


@nb.njit(cache=True)
def traversed_arcs(e_ptr, e_nodes, g_ptr, g_nbr, n):
    """Arcs traversed by the logged searches, as (parent, child) arrays, and the number of expanded records without one.

    The search inserts a record into its candidate list once, so the parent of an expanded record is the first earlier
    expanded record of the same sub-query whose out-list holds it. The first record of every list has no parent."""
    stamp = np.full(n, -1, np.int64); par = np.zeros(n, np.int32)
    parent = np.empty(len(e_nodes), np.int32); child = np.empty(len(e_nodes), np.int32); c = 0; unmatched = 0
    for q in range(len(e_ptr) - 1):
        for j in range(e_ptr[q], e_ptr[q + 1]):
            v = e_nodes[j]
            if j > e_ptr[q]:
                if stamp[v] == q:
                    parent[c] = par[v]; child[c] = v; c += 1
                else:
                    unmatched += 1
            stamp[v] = q
            for t in range(g_ptr[v], g_ptr[v + 1]):
                x = g_nbr[t]
                if stamp[x] != q:
                    stamp[x] = q; par[x] = v
    return parent[:c], child[:c], unmatched


@nb.njit(cache=True)
def in_lists(g_ptr, g_nbr, n):
    """In-lists of the graph as CSR: for every record the sources of its incoming arcs, in increasing order."""
    ptr = np.zeros(n + 1, np.int64)
    for t in range(len(g_nbr)):
        ptr[g_nbr[t] + 1] += 1
    for i in range(n):
        ptr[i + 1] += ptr[i]
    pos = ptr[:n].copy(); src = np.empty(len(g_nbr), np.int32)
    for u in range(n):
        for t in range(g_ptr[u], g_ptr[u + 1]):
            v = g_nbr[t]; src[pos[v]] = u; pos[v] += 1
    return ptr, src


@nb.njit(cache=True)
def _heat_rows(g_ptr, g_nbr, in_ptr, in_src, cached, structural, K, C, lo, hi, o2, n, ptr, nbr, wt, fill):
    """Rows of the heat graph, one record at a time.

    For record x it sums, per partner y, the structural arcs in both directions (when structural) and the traversal
    counts, then emits the partners above x and the partners below x in ascending order. With fill = False it only
    counts the entries per record into ptr[x + 1]; with fill = True it writes them from ptr[x]."""
    ys = np.empty(64, np.int64); ws = np.empty(64, np.int64); uy = np.empty(64, np.int64); uw = np.empty(64, np.int64); npairs = 0
    for x in range(n):
        m = 0; self_w = 0
        need = (g_ptr[x + 1] - g_ptr[x] + in_ptr[x + 1] - in_ptr[x] if structural else 0) + lo[x + 1] - lo[x] + hi[x + 1] - hi[x]
        if need > len(ys):
            ys = np.empty(2 * need, np.int64); ws = np.empty(2 * need, np.int64)
            uy = np.empty(2 * need, np.int64); uw = np.empty(2 * need, np.int64)
        if structural and not cached[x]:
            for t in range(g_ptr[x], g_ptr[x + 1]):
                y = g_nbr[t]
                if y != x and not cached[y]:
                    ys[m] = y; ws[m] = 1; m += 1
            for t in range(in_ptr[x], in_ptr[x + 1]):
                y = in_src[t]
                if y != x and not cached[y]:
                    ys[m] = y; ws[m] = 1; m += 1
        for j in range(lo[x], lo[x + 1]):
            y = K[j] % n
            if y == x:
                self_w += C[j]
            else:
                ys[m] = y; ws[m] = C[j]; m += 1
        for j in range(hi[x], hi[x + 1]):
            k = o2[j]; y = K[k] // n
            if y != x:
                ys[m] = y; ws[m] = C[k]; m += 1
        o = np.argsort(ys[:m], kind='mergesort')
        cnt = 0
        for i in range(m):
            y = ys[o[i]]
            if cnt > 0 and uy[cnt - 1] == y:
                uw[cnt - 1] += ws[o[i]]
            else:
                uy[cnt] = y; uw[cnt] = ws[o[i]]; cnt += 1
        tot = cnt + (2 if self_w > 0 else 0)
        for i in range(cnt):
            if uy[i] > x:
                npairs += 1
        if self_w > 0:
            npairs += 1
        if not fill:
            ptr[x + 1] = tot
            continue
        p = ptr[x]
        if self_w > 0:
            nbr[p] = x; wt[p] = self_w; p += 1
        for i in range(cnt):
            if uy[i] > x:
                nbr[p] = uy[i]; wt[p] = uw[i]; p += 1
        for i in range(cnt):
            if uy[i] < x:
                nbr[p] = uy[i]; wt[p] = uw[i]; p += 1
        if self_w > 0:
            nbr[p] = x; wt[p] = self_w; p += 1
    return npairs


def heat_graph(e_ptr, e_nodes, g_ptr, g_nbr, n, cached=None, structural=False):
    """Weighted graph of the HEAT baseline as symmetric CSR.

    w_uv sums, over the two directions of an arc between u and v, the number of logged traversals of the arc and, when
    structural is set, one more per arc. Arcs with an end in cached (a boolean mask or None) are dropped, which gives
    the weights on residual unions. Returns ((ptr, nbr, weight), number of pairs, counters)."""
    parent, child, unmatched = traversed_arcs(e_ptr.astype(np.int64), e_nodes.astype(np.int32), g_ptr, g_nbr, n)
    keep = None if cached is None else ~(cached[parent] | cached[child])
    k = np.minimum(parent, child).astype(np.int64) * n + np.maximum(parent, child)
    stats = dict(traversals=int(len(parent)), unrecovered=int(unmatched), dropped_cached=int(0 if keep is None else (~keep).sum()))
    if keep is not None:
        k = k[keep]
    del parent, child, keep
    # one key min * n + max per traversal; K are the unique pairs and C their traversal counts
    K, C = np.unique(k, return_counts=True); del k
    C = C.astype(np.int64)
    lo = np.zeros(n + 1, np.int64); lo[1:] = np.cumsum(np.bincount(K // n, minlength=n))
    mx = K % n; o2 = np.lexsort((K // n, mx)).astype(np.int64)
    hi = np.zeros(n + 1, np.int64); hi[1:] = np.cumsum(np.bincount(mx, minlength=n)); del mx
    cached_mask = np.zeros(n, np.bool_) if cached is None else cached.astype(np.bool_)
    if structural:
        in_ptr, in_src = in_lists(g_ptr, g_nbr, n)
    else:
        in_ptr, in_src = np.zeros(n + 1, np.int64), np.zeros(1, np.int32)
    ptr = np.zeros(n + 1, np.int64)
    _heat_rows(g_ptr, g_nbr, in_ptr, in_src, cached_mask, structural, K, C, lo, hi, o2, n, ptr, np.zeros(1, np.int32), np.zeros(1, np.int64), False)
    ptr = np.cumsum(ptr); nbr = np.empty(ptr[-1], np.int32); wt = np.empty(ptr[-1], np.int64)
    npairs = _heat_rows(g_ptr, g_nbr, in_ptr, in_src, cached_mask, structural, K, C, lo, hi, o2, n, ptr, nbr, wt, True)
    return (ptr, nbr, wt), npairs, stats


@nb.njit(cache=True)
def _sort_rows(ptr, idx):
    """Sort every CSR row in place."""
    for i in range(len(ptr) - 1):
        idx[ptr[i]:ptr[i + 1]] = np.sort(idx[ptr[i]:ptr[i + 1]])


@nb.njit(cache=True)
def _transpose(g_ptr, g_nbr, n):
    """In-lists of a graph whose out-lists are sorted; the in-lists come out sorted too."""
    cnt = np.zeros(n + 1, np.int64)
    for j in range(len(g_nbr)):
        cnt[g_nbr[j] + 1] += 1
    in_ptr = np.cumsum(cnt); pos = in_ptr[:-1].copy(); in_src = np.empty(len(g_nbr), np.int32)
    for u in range(n):
        for j in range(g_ptr[u], g_ptr[u + 1]):
            v = g_nbr[j]; in_src[pos[v]] = u; pos[v] += 1
    return in_ptr, in_src


@nb.njit(cache=True)
def _merge(g_ptr, g_sorted, in_ptr, in_src, n, ptr, vv, ww, fill):
    """Merge the sorted out- and in-list of every record into one row without duplicates.

    The weight of a partner is the number of directions in which the arc exists. Partners above the record come first,
    then partners below it. With fill = False only the row lengths are written to ptr."""
    for u in range(n):
        a, ae, b, be = g_ptr[u], g_ptr[u + 1], in_ptr[u], in_ptr[u + 1]; k = ptr[u] if fill else 0; c = 0
        for side in range(2):
            i, j = a, b
            lastv = -1
            while i < ae or j < be:
                x = g_sorted[i] if i < ae else 2147483647; y = in_src[j] if j < be else 2147483647
                v = x if x <= y else y; w = 0
                if x == v:
                    w += 1; i += 1
                if y == v:
                    w += 1; j += 1
                if v == u or v == lastv:
                    continue
                if (side == 0 and v > u) or (side == 1 and v < u):
                    if fill:
                        vv[k] = v; ww[k] = w; k += 1
                    else:
                        c += 1
                lastv = v
        if not fill:
            ptr[u + 1] = c


def starling_graph(g_ptr, g_nbr, n):
    """Graph of the STARLING baseline as symmetric CSR: w_uv = number of directions in which the index graph has the arc {u, v}.

    A layout that maximizes this weight inside pages maximizes the number of in-page arcs."""
    g_sorted = g_nbr.astype(np.int32).copy(); _sort_rows(g_ptr, g_sorted)
    in_ptr, in_src = _transpose(g_ptr, g_sorted, n)
    ptr = np.zeros(n + 1, np.int64); _merge(g_ptr, g_sorted, in_ptr, in_src, n, ptr, np.zeros(1, np.int32), np.zeros(1, np.int64), False)
    ptr = np.cumsum(ptr); vv = np.empty(ptr[-1], np.int32); ww = np.empty(ptr[-1], np.int64)
    _merge(g_ptr, g_sorted, in_ptr, in_src, n, ptr, vv, ww, True)
    return ptr, vv, ww
