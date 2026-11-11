"""Layout objectives over a scope hypergraph: connectivity, clique, probabilistic fanout and pairwise co-location."""
import numpy as np
import numba as nb

# [Paper] Section VI-B. SET minimizes F_conn (Eq. 1), CLQ and PAIRWISE maximize the pairwise objectives of
# Definition 9 (CLQ on all co-member pairs, PAIRWISE on an explicit weighted graph such as the arc-restricted pairs),
# and P05 minimizes the probabilistic fanout at theta = 1/2.
SET, CLQ, PAIRWISE, P05 = 0, 1, 2, 3


def fanout_table():
    """Integer table of the probabilistic fanout at theta = 1/2, scaled by 2^12.

    fan[c] = 2^12 * (1 - 2^-c) for c = 0..12 and fan_step[c] = fan[c + 1] - fan[c]. Every entry is an integer for
    c <= 12, so the gains of P05 are exact for page capacities B <= 12."""
    num, den = 1, 2
    fan = np.array([den ** 12 - num ** c * den ** (12 - c) for c in range(13)], np.int64)
    fan_step = np.zeros(13, np.int64); fan_step[:12] = fan[1:] - fan[:-1]
    return fan, fan_step


@nb.njit(cache=True)
def connectivity_and_clique(e_ptr, e_nodes, page, npages):
    """Set statistics of a layout over the hyperedges (e_ptr, e_nodes), each with weight 1.

    Returns (conn, clique, bonus, base):
      conn   = sum_E lambda_pi(E), the number of distinct pages of every set (F_conn + number of sets);
      clique = sum_E sum_p C(c_{E,p}, 2), the co-located pairs counted by the clique objective;
      bonus  = sum_E sum_p C(c_{E,p} - 1, 2), the concentration bonus;
      base   = sum_E (|E| - 1).
    They satisfy clique = (base + m - conn) + bonus for m sets."""
    m = len(e_ptr) - 1
    cnt = np.zeros(npages, np.int64)
    conn = 0; clique = 0; bonus = 0; base = 0
    for e in range(m):
        for j in range(e_ptr[e], e_ptr[e + 1]):
            cnt[page[e_nodes[j]]] += 1
        for j in range(e_ptr[e], e_ptr[e + 1]):
            p = page[e_nodes[j]]
            c = cnt[p]
            # [Paper] Definition 2: c is the page count c_{E,p}; each page of the set is visited once
            if c > 0:
                conn += 1
                clique += c * (c - 1) // 2
                bonus += (c - 1) * (c - 2) // 2
                cnt[p] = 0
        base += e_ptr[e + 1] - e_ptr[e] - 1
    return conn, clique, bonus, base


@nb.njit(cache=True)
def fanout_value(e_ptr, e_nodes, page, npages, fan):
    """Probabilistic fanout sum_E sum_{p in pi(E)} fan[c_{E,p}] (scaled by 2^12)."""
    m = len(e_ptr) - 1
    cnt = np.zeros(npages, np.int64)
    s = 0
    for e in range(m):
        for j in range(e_ptr[e], e_ptr[e + 1]):
            cnt[page[e_nodes[j]]] += 1
        for j in range(e_ptr[e], e_ptr[e + 1]):
            p = page[e_nodes[j]]
            c = cnt[p]
            if c > 0:
                s += fan[c]
                cnt[p] = 0
    return s


@nb.njit(cache=True)
def pairwise_value(w_ptr, w_nbr, w_val, page):
    """Pairwise co-location value sum_{u<v} w_uv [pi(u) = pi(v)] of a symmetric weighted graph in CSR form."""
    n = len(w_ptr) - 1
    s = 0
    for u in range(n):
        pu = page[u]
        for t in range(w_ptr[u], w_ptr[u + 1]):
            if page[w_nbr[t]] == pu:
                s += w_val[t]
    return s // 2


def objective_value(mode, e_ptr, e_nodes, w_ptr, w_nbr, w_val, page, npages, fan):
    """Value that refinement minimizes for one objective.

    SET returns sum_E lambda_pi(E), P05 the scaled probabilistic fanout. CLQ and PAIRWISE are maximized in the paper, so
    their negated values are returned."""
    if mode == SET:
        return connectivity_and_clique(e_ptr, e_nodes, page, npages)[0]
    if mode == CLQ:
        return -connectivity_and_clique(e_ptr, e_nodes, page, npages)[1]
    if mode == P05:
        return fanout_value(e_ptr, e_nodes, page, npages, fan)
    return -pairwise_value(w_ptr, w_nbr, w_val, page)
