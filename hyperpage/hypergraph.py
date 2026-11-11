"""Scope hypergraphs built from logged expanded sets: per-query sets, scope unions and residual unions."""
import numpy as np
import numba as nb


@nb.njit(cache=True)
def per_query_sets(queries, q_ptr, q_nodes):
    """Per-query hypergraph: one set e_q per listed sub-query, in the given order.

    (q_ptr, q_nodes) is the CSR of the logged expanded sets and queries indexes into it."""
    tot = 0
    for c in queries:
        tot += q_ptr[c + 1] - q_ptr[c]
    e_ptr = np.zeros(len(queries) + 1, np.int64); e_nodes = np.empty(tot, np.int32); k = 0
    for i, c in enumerate(queries):
        for j in range(q_ptr[c], q_ptr[c + 1]):
            e_nodes[k] = q_nodes[j]; k += 1
        e_ptr[i + 1] = k
    return e_ptr, e_nodes


@nb.njit(cache=True)
def scope_unions(s_ptr, queries, q_ptr, q_nodes, n):
    """Scope hypergraph: one set E_sigma per scope, the union of the expanded sets of its sub-queries.

    Scope i holds the sub-queries queries[s_ptr[i]:s_ptr[i + 1]]. Records appear in first-occurrence order.
    Identical unions stay separate sets of weight 1, which gives every objective the value it has when they are
    merged with their multiplicity."""
    # [Paper] Definition 8: E_sigma is the union of e_q over the sub-queries q of scope sigma
    stamp = np.full(n, -1, np.int64)
    tot = 0
    for b in range(len(s_ptr) - 1):
        for t in range(s_ptr[b], s_ptr[b + 1]):
            c = queries[t]
            for j in range(q_ptr[c], q_ptr[c + 1]):
                v = q_nodes[j]
                if stamp[v] != b:
                    stamp[v] = b; tot += 1
    stamp[:] = -1
    e_ptr = np.zeros(len(s_ptr), np.int64); e_nodes = np.empty(tot, np.int32); k = 0
    for b in range(len(s_ptr) - 1):
        for t in range(s_ptr[b], s_ptr[b + 1]):
            c = queries[t]
            for j in range(q_ptr[c], q_ptr[c + 1]):
                v = q_nodes[j]
                if stamp[v] != b:
                    stamp[v] = b; e_nodes[k] = v; k += 1
        e_ptr[b + 1] = k
    return e_ptr, e_nodes


def incidence(set_ptr, set_nodes, n):
    """Incidence lists of a hypergraph given as CSR over its sets.

    Returns (e_ptr, e_nodes, v_ptr, v_edges, v_pos): the sets, and for every record the sets that contain it
    (v_edges, ascending) with the position of the record inside each set (v_pos, in expansion order)."""
    e_ptr = set_ptr.astype(np.int64)
    e_nodes = set_nodes.astype(np.int32)
    m = len(e_ptr) - 1
    set_id = np.repeat(np.arange(m, dtype=np.int32), np.diff(e_ptr))
    o = np.lexsort((set_id, e_nodes))
    v_edges = set_id[o]
    cnt = np.bincount(e_nodes, minlength=n)
    v_ptr = np.zeros(n + 1, np.int64); v_ptr[1:] = np.cumsum(cnt)
    v_pos = (o - e_ptr[set_id[o]]).astype(np.int64)
    return e_ptr, e_nodes, v_ptr, v_edges, v_pos


def residual_sets(e_ptr, e_nodes, cached):
    """Residual unions E \\ C: every set without the records of the cache C (cached is a boolean mask over records).

    Sets keep their index, so a set may become empty.
    """
    # [Paper] Corollary 3 and Section VI-D: cached records add no reads, so refinement runs on E \\ C
    keep = ~cached[e_nodes]; sizes = np.diff(e_ptr)
    set_id = np.repeat(np.arange(len(sizes)), sizes)[keep]
    ptr = np.concatenate([[0], np.cumsum(np.bincount(set_id, minlength=len(sizes)))]).astype(np.int64)
    return ptr, e_nodes[keep]


def most_frequent_records(e_ptr, e_nodes, n, count):
    """Mask of the count records that occur in the most sets, ties broken by the smaller identifier."""
    # [Paper] Definition 7: the record cache holds the records most frequent in training
    freq = np.bincount(e_nodes, minlength=n); mask = np.zeros(n, bool)
    mask[np.lexsort((np.arange(n), -freq))[:count]] = True
    return mask
