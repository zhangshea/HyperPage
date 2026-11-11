"""End-to-end check on a small synthetic workload: hypergraphs, Algorithm 1 under SET, CLQ, ARC, P05 and HEAT,
cache-aware refinement, replay under the hybrid cache and selection."""
import numpy as np

from hyperpage import cache, objectives, selection
from hyperpage.hypergraph import scope_unions, per_query_sets, incidence
from hyperpage.layout import refine_layout

N, B, DEGREE = 2400, 12, 8


def synthetic_graph(rng):
    """Out-lists of a graph whose arcs join records with close identifiers; no self-loops."""
    lists = [np.setdiff1d(np.unique((v + rng.integers(-40, 40, DEGREE)) % N), [v]) for v in range(N)]
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum([len(x) for x in lists])
    return ptr, np.concatenate(lists).astype(np.int32)


def synthetic_log(rng, n_queries):
    """Expanded sets of sub-queries: each walks a neighbourhood of a random record, in expansion order."""
    sets = []
    for _ in range(n_queries):
        x = (rng.integers(0, N) + rng.integers(-70, 70, rng.integers(8, 30))) % N
        _, first = np.unique(x, return_index=True)
        sets.append(x[np.sort(first)])
    ptr = np.zeros(n_queries + 1, np.int64)
    ptr[1:] = np.cumsum([len(x) for x in sets])
    return ptr, np.concatenate(sets).astype(np.int32)


def main():
    rng = np.random.default_rng(1)
    graph = synthetic_graph(rng)
    n_pages = -(-N // B)
    incumbent = (rng.permutation(N) // B).astype(np.int32)

    # training log: groups of 6 sub-queries; scope hypergraph = one union per group
    q_ptr, q_nodes = synthetic_log(rng, 600)
    group_ptr = np.arange(0, 601, 6).astype(np.int64)
    queries = np.arange(600)
    unions = scope_unions(group_ptr, queries, q_ptr, q_nodes, N)
    per_query = per_query_sets(queries, q_ptr, q_nodes)
    assert len(unions[0]) - 1 == 100 and len(per_query[0]) - 1 == 600

    # Proposition 9: refinement never increases its objective
    layouts = {'MARGO': incumbent}
    for name in ('SET', 'CLQ', 'ARC', 'P05', 'HEAT'):
        page, info = refine_layout(name, unions, graph, incumbent, B, seed=0, query_sets=per_query)
        assert info['value_end'] <= info['value_start'], name
        assert np.bincount(page, minlength=n_pages).max() <= B
        layouts[name + '-G'] = page
    # cache-aware refinement on the residual unions of a record cache of 10 page slots
    page, info = refine_layout('ARC', unions, graph, incumbent, B, seed=0, cached_records=10 * B)
    assert info['cached_records'] == 10 * B
    layouts['ARC-C'] = page

    # Corollary 1: reads of a read-once scope = connectivity of its union; refinement under SET lowers their sum
    e_ptr, e_nodes, _, _, _ = incidence(unions[0], unions[1], N)
    before = objectives.connectivity_and_clique(e_ptr, e_nodes, incumbent, n_pages)[0]
    after = objectives.connectivity_and_clique(e_ptr, e_nodes, layouts['SET-G'], n_pages)[0]
    assert after < before

    # replay of a held-out log under the hybrid cache: 12 steps, turn lockstep
    v_ptr, v_nodes = synthetic_log(np.random.default_rng(2), 480)
    step = np.repeat(np.arange(12), 40).astype(np.int64)
    turn = np.tile(np.repeat(np.arange(4), 10), 12).astype(np.int64)
    rollout = np.tile(np.arange(10), 48).astype(np.int64)
    position = np.zeros(480, np.int64)
    order = cache.lockstep_order(step, turn, rollout, position)
    freq = cache.record_frequency(unions[0], unions[1], N)
    ranking = cache.rank_records(freq)
    budget, rows = 20, {}
    for name, page in layouts.items():
        for share in cache.SHARE_GRID:
            hot_pages, cached_records, lru_pages = cache.split_budget(budget, share, B)
            mask = cache.cached_mask(page, n_pages, ranking, freq, hot_pages, cached_records)
            misses = cache.hybrid_cache_misses(order, v_ptr, v_nodes, page, n_pages, lru_pages, mask, step, 12)
            rows[selection.row_key(name, 0, budget, share)] = dict(per_step=misses.astype(float).tolist())
    # at share 1 the page LRU is empty, so every layout reads the same
    full = {selection.total_reads(rows, name, 0, budget, 1.0) for name in layouts}
    assert len(full) == 1

    # selection: the deployed pair never reads more validation pages than the incumbent at its best share
    candidates = [name for name in layouts if name != 'MARGO']
    layout, share, reads = selection.deployment(rows, candidates, 'MARGO', 0, budget, cache.SHARE_GRID, warmup=2, scored=10)
    best_incumbent = min(selection.scored_reads(rows, 'MARGO', 0, budget, s, 2, 10) for s in cache.SHARE_GRID)
    assert reads <= best_incumbent
    print(f'deployed {layout} at share {share}: {reads:.0f} validation reads, incumbent {best_incumbent:.0f}')
    print('synthetic checks passed')


if __name__ == '__main__':
    main()
