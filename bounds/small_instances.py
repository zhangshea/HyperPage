"""Small instances drawn from the test groups: exact optimum of the reads and the bound of Proposition 7 with one class
per record, solved as one linear program."""
# [Paper] Section VII-D: 71 small W3 instances solved to proven optimality (12 to 48 records, B from 2 to 12); the bound
# with one class per record reaches a median of 96% of the optimum at B = 3 and 81% to 90% at B = 12.
#
# Instance: a seed group E0 drawn uniformly from the groups with at least n records, n of its records drawn uniformly
# without replacement, and the projections onto those records of M groups (E0 and M - 1 others drawn uniformly).
# Projections with fewer than two selected records are dropped from the instance (an empty projection reads nothing, a
# one-record projection one page under every layout), so the optimum and the bound both exclude their reads; identical
# projections are merged, with their multiplicity as weight w_E.
# Optimum: min_pi sum_E w_E lambda_pi(E) over layouts with B records on each of k = n / B pages, by CP-SAT with
#   x[v, p] (record v on page p) and y[E, p] (set E touches page p), y >= x for v in E, and record v only on pages
#   p <= v, which removes page symmetry (pages can always be ordered by their smallest record).
# B = 2: the optimum is also computed by maximum-weight perfect matching (Corollary 2).
# Bound: sup over (gamma, mu) of Proposition 7 for a fixed class partition. The inner maximum over |T| <= beta is
#   replaced by its dual min_{theta >= 0} beta theta + sum_u max(0, a_u - theta) and psi_E by z <= h(t) + gamma t for
#   t = 0..m_E, which turns the supremum into one linear program. Its optimum is re-evaluated with the formula of
#   Proposition 7 at the returned (gamma, mu) as a check.
import argparse
import json
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def draw_instance(rng, e_ptr, e_nodes, n, M):
    """One instance; the random calls are made in a fixed order, so a seed fixes the whole sequence of instances."""
    G = len(e_ptr) - 1
    g0 = int(rng.integers(G)); E0 = e_nodes[e_ptr[g0]:e_ptr[g0 + 1]]
    while len(E0) < n:   # groups with fewer than n records are redrawn
        g0 = int(rng.integers(G)); E0 = e_nodes[e_ptr[g0]:e_ptr[g0 + 1]]
    S = np.sort(rng.choice(E0, n, replace=False)); loc = {int(v): i for i, v in enumerate(S)}
    gs = np.concatenate([[g0], rng.choice(np.setdiff1d(np.arange(G), [g0]), M - 1, replace=False)])
    cnt = {}
    for g in gs:
        pr = tuple(sorted(loc[int(v)] for v in e_nodes[e_ptr[g]:e_ptr[g + 1]] if int(v) in loc))
        if len(pr) >= 2:
            cnt[pr] = cnt.get(pr, 0) + 1
    sets = sorted(cnt)
    return dict(g0=g0, records=S.tolist(), sets=[list(s) for s in sets], w=[cnt[s] for s in sets])


def exact_optimum(sets, w, n, B, time_limit, workers):
    """CP-SAT model of the minimum weighted reads over layouts with B records per page."""
    from ortools.sat.python import cp_model
    k = n // B
    m = cp_model.CpModel(); x = [[m.NewBoolVar('') if p <= v else None for p in range(k)] for v in range(n)]
    for v in range(n):
        m.AddExactlyOne([x[v][p] for p in range(min(v, k - 1) + 1)])
    for p in range(k):
        m.Add(sum(x[v][p] for v in range(p, n)) <= B)
    obj = []
    for E, we in zip(sets, w):
        for p in range(k):
            pins = [x[v][p] for v in E if p <= v]
            if not pins:
                continue
            y = m.NewBoolVar('')
            for xv in pins:
                m.AddImplication(xv, y)
            obj.append(we * y)
    m.Minimize(sum(obj)); s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit; s.parameters.num_workers = workers
    st = s.Solve(m)
    page = None
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        page = [next(p for p in range(k) if x[v][p] is not None and s.Value(x[v][p])) for v in range(n)]
    return dict(status=s.StatusName(st), value=s.ObjectiveValue(), bound=s.BestObjectiveBound(), secs=s.WallTime(),
                page=page)


def matching_optimum(sets, w, n):
    """B = 2 (Corollary 2): reads = sum_E w_E |E| minus a maximum-weight perfect matching under
    w_uv = sum_{E containing u, v} w_E."""
    # [Paper] Corollary 2
    import networkx as nx
    g = nx.Graph(); g.add_nodes_from(range(n))
    for E, we in zip(sets, w):
        for i, u in enumerate(E):
            for v in E[i + 1:]:
                g.add_edge(u, v, weight=g.get_edge_data(u, v, {'weight': 0})['weight'] + we)
    for u in range(n):   # zero-weight edges keep a perfect matching feasible
        for v in range(u + 1, n):
            if not g.has_edge(u, v):
                g.add_edge(u, v, weight=0)
    M_ = nx.max_weight_matching(g, maxcardinality=True); assert len(M_) == n // 2
    page = [0] * n
    for p, (u, v) in enumerate(M_):
        page[u] = page[v] = p
    return float(sum(we * len(E) for E, we in zip(sets, w)) - sum(g[u][v]['weight'] for u, v in M_)), page


def weighted_reads(sets, w, page):
    return float(sum(we * len({page[v] for v in E}) for E, we in zip(sets, w)))


def lp_bound(sets, w, classes, n, B):
    """max sum_{v,E} w_E z_vE - sum_v beta theta_v - sum_(v,u) s_vu - beta sum_i mu_i |C_i| over gamma, z (free) per
    incidence and theta_v, s_vu, mu_i >= 0, subject to z_vE - t gamma_vE <= 1/(1+t) for t = 0..m_E and
    sum_{E containing u, v} w_E gamma_vE - mu_c(u) - theta_v - s_vu <= 0."""
    # [Paper] Proposition 7 as a linear program over (gamma, mu)
    beta = B - 1
    inc = [(v, j) for j, E in enumerate(sets) for v in E]; ni = len(inc); idx = {vj: i for i, vj in enumerate(inc)}
    pairs = sorted({(v, u) for E in sets for v in E for u in E if u != v}); npair = len(pairs)
    pid = {p: i for i, p in enumerate(pairs)}
    cls = np.array(classes); r = int(cls.max()) + 1; csz = np.bincount(cls, minlength=r)
    # columns: gamma [0, ni), z [ni, 2ni), theta [2ni, 2ni+n), s [.., +npair), mu [.., +r)
    og, oz, ot, os_, om = 0, ni, 2 * ni, 2 * ni + n, 2 * ni + n + npair; nc = om + r
    cost = np.zeros(nc); lo = np.zeros(nc); lo[og:ot] = -np.inf
    for i, (v, j) in enumerate(inc):
        cost[oz + i] = w[j]
    cost[ot:os_] = -beta; cost[os_:om] = -1.0; cost[om:] = -beta * csz
    R, C, V, rhs = [], [], [], []; row = 0
    for i, (v, j) in enumerate(inc):
        for t in range(min(beta, len(sets[j]) - 1) + 1):
            R += [row, row]; C += [oz + i, og + i]; V += [1.0, -float(t)]; rhs.append(1.0 / (1 + t)); row += 1
    cols = {}
    for j, E in enumerate(sets):
        for v in E:
            for u in E:
                if u != v:
                    cols.setdefault(pid[(v, u)], []).append((og + idx[(v, j)], float(w[j])))
    for q, (v, u) in enumerate(pairs):
        for c, val in cols[q]:
            R.append(row); C.append(c); V.append(val)
        R += [row, row, row]; C += [om + cls[u], ot + v, os_ + q]; V += [-1.0, -1.0, -1.0]; rhs.append(0.0); row += 1
    A = coo_matrix((V, (R, C)), shape=(row, nc)).tocsr()
    # SciPy's bundled HiGHS (interior point); a separately installed HiGHS library conflicts with OR-Tools' copy
    t0 = time.time()
    L = linprog(-cost, A_ub=A, b_ub=np.array(rhs), bounds=list(zip(lo, [None] * nc)), method='highs-ipm')
    sol = L.x; val = -L.fun
    g = sol[og:oz]; mu = np.maximum(sol[om:], 0.0)
    # check: Proposition 7 at (gamma, mu), sum psi - sum_v max_{|T|<=beta} sum (w_vu - mu_c(u)) - beta sum mu |C|
    ev = sum(w[j] * min(1.0 / (1 + t) + g[i] * t for t in range(min(beta, len(sets[j]) - 1) + 1))
             for i, (v, j) in enumerate(inc))
    wg = {}
    for j, E in enumerate(sets):
        for v in E:
            for u in E:
                if u != v:
                    wg[(v, u)] = wg.get((v, u), 0.0) + w[j] * g[idx[(v, j)]]
    sc = {}
    for (v, u), a in wg.items():
        sc.setdefault(v, []).append(a - mu[cls[u]])
    ev -= sum(sum(sorted([a for a in sc.get(v, []) if a > 0], reverse=True)[:beta]) for v in range(n))
    ev -= beta * float((mu * csz).sum())
    return dict(value=float(val), recheck=float(ev), status=L.message, secs=time.time() - t0, rows=row, cols=nc)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--unions', required=True, help='.npz with e_ptr, e_nodes of the test group unions')
    ap.add_argument('--out', required=True, help='output JSON (rewritten after every instance)')
    ap.add_argument('--records', type=int, required=True, help='records per instance n (a multiple of B)')
    ap.add_argument('--B', type=int, required=True, help='page capacity (records per page)')
    ap.add_argument('--seed', type=int, default=764575224, help='random seed (the same for every configuration)')
    ap.add_argument('--count', type=int, default=10, help='instances per configuration')
    ap.add_argument('--groups', type=int, default=2000, help='groups M projected onto the records of an instance')
    ap.add_argument('--time-limit', type=float, default=600.0, help='CP-SAT time limit per instance in seconds')
    ap.add_argument('--workers', type=int, default=8, help='CP-SAT worker threads')
    A = ap.parse_args()
    n, B = A.records, A.B; k = n // B; assert k * B == n, 'the number of records must be a multiple of B'
    U = np.load(A.unions); e_ptr, e_nodes = U['e_ptr'].astype(np.int64), U['e_nodes'].astype(np.int64)
    rng = np.random.default_rng(A.seed)
    res = dict(seed=A.seed, n=n, B=B, k=k, M=A.groups, time_limit=A.time_limit, instances=[])
    for it in range(A.count):
        I = draw_instance(rng, e_ptr, e_nodes, n, A.groups); sets, w = I['sets'], I['w']
        O = exact_optimum(sets, w, n, B, A.time_limit, A.workers)
        row = dict(g0=I['g0'], n_sets=len(sets), w_sum=float(sum(w)), opt=O['value'], opt_bound=O['bound'],
                   opt_status=O['status'], opt_secs=O['secs'])
        if O['page'] is not None:
            row['opt_recheck'] = weighted_reads(sets, w, O['page'])
        if B == 2:
            mv, mpage = matching_optimum(sets, w, n)
            row['opt_matching'] = mv; row['opt_matching_recheck'] = weighted_reads(sets, w, mpage)
        row['lb_single'] = lp_bound(sets, w, list(range(n)), n, B)
        res['instances'].append(row); json.dump(res, open(A.out, 'w'), indent=1)
        short = {kk: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a in ('value', 'recheck')})
                 for kk, v in row.items()}
        print(json.dumps(short), flush=True)


if __name__ == '__main__':
    main()
