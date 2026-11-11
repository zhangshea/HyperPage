"""Construction of one refined layout: objective set-up, residual unions for a record cache, and Algorithm 1."""
import time
import numpy as np

from . import objectives as obj
from .hypergraph import incidence, residual_sets, most_frequent_records
from .refine import build_count_tables, slots_of_layout, refine
from .weights import arc_pairs, pairs_to_csr, heat_graph, starling_graph

OBJECTIVES = ('SET', 'CLQ', 'ARC', 'P05', 'HEAT', 'STARLING')


def refine_layout(objective, sets, graph, start, capacity, seed, max_passes=8, tol=1e-4, max_cand=16, max_sets=64,
                  window=32, cached_records=0, freeze_uncovered=False, query_sets=None, heat_arc_term=True):
    """Refine a start layout under one objective and return (layout, info).

    objective        one of OBJECTIVES. SET, CLQ, ARC and P05 use the sets; HEAT uses the per-query sets in
                     query_sets and the graph; STARLING uses the graph alone.
    sets             (e_ptr, e_nodes) of the training hypergraph (group unions or per-query sets).
    graph            (g_ptr, g_nbr), out-neighbour lists of the index graph.
    start            the incumbent layout, page[v] for every record v.
    capacity         records per page, B.
    seed             seed of the random order in which records are visited.
    cached_records   number of most frequent training records that the record cache will hold; refinement then runs
                     on the residual unions (cache-aware refinement). 0 gives cache-blind refinement.
    freeze_uncovered keep the records outside the training sets on their start pages (they are no swap partners).
    heat_arc_term    HEAT only: add one to the weight of every graph arc in each direction.
    """
    assert objective in OBJECTIVES, objective
    e_ptr, e_nodes = sets
    g_ptr, g_nbr = graph
    n = len(start)
    uncovered = np.bincount(e_nodes, minlength=n) == 0

    # [Paper] Section VI-D: the record cache holds the most frequent training records, and cache-aware refinement
    # runs Algorithm 1 on the residual unions E \ C.
    cached = None
    if cached_records > 0:
        cached = most_frequent_records(e_ptr, e_nodes, n, cached_records)
        e_ptr, e_nodes = residual_sets(e_ptr, e_nodes, cached)
    e_ptr, e_nodes, v_ptr, v_edges, v_pos = incidence(e_ptr, e_nodes, n)
    n_sets = len(e_ptr) - 1
    degree = np.diff(v_ptr)

    frozen = np.zeros(n, bool)
    if freeze_uncovered:
        # with a record cache, cached records stay movable: their slots serve as free partners for swaps
        frozen |= uncovered if cached_records > 0 else (degree == 0)

    t0 = time.time()
    empty_graph = (np.zeros(n + 1, np.int64), np.zeros(1, np.int32), np.zeros(1, np.int64))
    fan = fan_step = np.zeros(13, np.int64)
    weights = empty_graph
    if objective == 'SET':
        mode = obj.SET
    elif objective == 'CLQ':
        mode = obj.CLQ
    elif objective == 'P05':
        assert capacity <= 12, 'the integer fanout table covers page capacities up to 12'
        mode = obj.P05
        fan, fan_step = obj.fanout_table()
    elif objective == 'ARC':
        mode = obj.PAIRWISE
        weights = pairs_to_csr(*arc_pairs(e_ptr, e_nodes, g_ptr, g_nbr, n), n)
    elif objective == 'HEAT':
        assert query_sets is not None, 'HEAT needs the per-query training sets'
        mode = obj.PAIRWISE
        weights, _, _ = heat_graph(query_sets[0], query_sets[1], g_ptr, g_nbr, n, cached=cached, structural=heat_arc_term)
    else:
        mode = obj.PAIRWISE
        weights = starling_graph(g_ptr, g_nbr, n)

    # [Paper] Section VI-C: candidate pages also include the pages of graph neighbours inside training sets
    if objective in ('ARC', 'STARLING'):
        adjacency = weights[:2]
    else:
        adjacency = pairs_to_csr(*arc_pairs(e_ptr, e_nodes, g_ptr, g_nbr, n), n)[:2]

    # [Paper] Algorithm 1, line 1: the records that initiate operations, in a random order. These are the members of
    # the (residual) training sets or, for a pairwise objective, the records with a weighted partner.
    if mode == obj.PAIRWISE:
        covered = np.nonzero(np.diff(weights[0]) > 0)[0].astype(np.int32)
    else:
        covered = np.nonzero(degree > 0)[0].astype(np.int32)
    order = covered[np.random.default_rng(seed).permutation(len(covered))]

    n_pages = -(-n // capacity)
    page = start.astype(np.int32).copy()
    value0 = obj.objective_value(mode, e_ptr, e_nodes, weights[0], weights[1], weights[2], page, n_pages, fan)
    slots = slots_of_layout(page, n_pages, capacity)
    if mode != obj.PAIRWISE:
        t_ptr, t_keys, t_vals = build_count_tables(e_ptr, e_nodes, page)
    else:
        t_ptr, t_keys, t_vals = np.zeros(1, np.int64), np.zeros(1, np.int32), np.zeros(1, np.int32)
    prepare_seconds = time.time() - t0

    t1 = time.time()
    value, passes, applied, pass_gain = refine(
        mode, page, slots, capacity, n_pages, e_ptr, e_nodes, v_ptr, v_edges, t_ptr, t_keys, t_vals,
        weights[0], weights[1], weights[2], order, max_cand, max_passes, tol, value0, frozen, v_pos, max_sets, window,
        adjacency[0], adjacency[1], fan_step)
    refine_seconds = time.time() - t1

    # the incrementally maintained value must equal the value recomputed from the final layout
    check = obj.objective_value(mode, e_ptr, e_nodes, weights[0], weights[1], weights[2], page, n_pages, fan)
    assert check == value, (objective, check, value)
    assert np.bincount(page, minlength=n_pages).max() <= capacity
    conn, clique, bonus, base = obj.connectivity_and_clique(e_ptr, e_nodes, page, n_pages)
    # [Paper] Proposition 1: clique credit = set credit + concentration bonus
    assert clique == (base + n_sets - conn) + bonus
    info = dict(objective=objective, capacity=int(capacity), pages=int(n_pages), sets=int(n_sets), incidences=int(len(e_nodes)),
                coverage=float((degree > 0).mean()), cached_records=int(cached_records), frozen_records=int(frozen.sum()),
                seed=int(seed), max_passes=int(max_passes), passes=int(passes), applied_operations=int(applied),
                pass_gain=[int(x) for x in pass_gain[:passes]], value_start=int(value0), value_end=int(value),
                connectivity_sum=int(conn), moved_records=int((page != start).sum()),
                prepare_seconds=prepare_seconds, refine_seconds=refine_seconds)
    return page, info
