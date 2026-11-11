"""Lagrangian lower bound of Proposition 7 on the reads of every layout, with one refinement of the record classes.

Reads unit-weight sets (e_ptr, e_nodes) and writes the best bound value per set, its history and the best multipliers.
"""
# [Paper] Proposition 7 and Section VII-D: 543.2 reads per W3-1M test group with 67 record classes.
#
# Notation (Proposition 7): beta = B - 1, h(t) = 1 / (1 + t), m_E = min(beta, |E| - 1),
# psi_E(g) = min_{t in 0..m_E} h(t) + g t. One pin x = (v, E) per record v and set E containing v carries gamma_x.
# The bound is
#   L(gamma, mu) = sum_x psi_E(gamma_x) - sum_v max_{|T| <= beta} sum_{u in T} (w_vu - mu_c(u)) - beta sum_i mu_i |C_i|,
#   w_vu = sum_{E containing u, v} gamma_x.
# The pin multipliers are parametrised as gamma_x + kappa_v with kappa_v >= 0 chosen per record (record_blocks). The
# kappa part of the inner maximum is bounded separately: with c_vu the number of sets holding u and v,
#   max_{|T| <= beta} sum_{u in T} (w_vu + kappa_v c_vu - mu_c(u)) <= max_{|T| <= beta} sum_{u in T} (w_vu - mu_c(u))
#                                                                    + kappa_v top_beta(v),
# where top_beta(v) is the sum of the beta largest c_vu. Every evaluated value is therefore a valid lower bound obtained
# from Proposition 7 at the multipliers gamma + kappa by this further relaxation; it can lie below Proposition 7's value
# at those multipliers. The run has three stages:
#   class stage: a class-capacity bound on degree classes gives the starting gamma (class_bound);
#   stage A: outer iterations that optimise mu at fixed gamma (ascend_prices) and move gamma toward the
#            subdifferential at the averaged page-mate counts (project_to_average);
#   split:   every class is split by the averaged in-load of its records (split_classes), children inherit the price
#            of their parent, which keeps the value; the first evaluation after the split must reproduce it;
#   stage B: further outer iterations on the refined classes.
import argparse
import json
import sys
import time

import numpy as np
import numba as nb

from hyperpage.hypergraph import incidence


_T0 = time.time()


def log(*a):
    print(f'[{time.time() - _T0:8.0f} s]', *a, flush=True)


@nb.njit(parallel=True, cache=True)
def sort_set_members(e_ptr, e_nodes):
    for e in nb.prange(len(e_ptr) - 1):
        e_nodes[e_ptr[e]:e_ptr[e + 1]] = np.sort(e_nodes[e_ptr[e]:e_ptr[e + 1]])


@nb.njit(parallel=True, cache=True)
def class_partner_scores(e_ptr, e_nodes, v_ptr, v_sets, n, cls, r, b, gam, nthreads):
    """Per record v: score[v, i, j] and partner[v, i, j], the j-th best partner of v in class i under w_vu
    (gamma weights); top_beta[v], the sum of the b largest plain co-occurrence counts of v."""
    score = np.zeros((n, r, b)); partner = np.full((n, r, b), -1, np.int32); top_beta = np.zeros(n)
    for t in nb.prange(nthreads):
        acc = np.zeros(n); cnt = np.zeros(n, np.int32); touched = np.empty(n, np.int32)
        for v in range(t, n, nthreads):
            nt = 0
            for x in range(v_ptr[v], v_ptr[v + 1]):
                e = v_sets[x]; g = gam[x]
                for j in range(e_ptr[e], e_ptr[e + 1]):
                    u = e_nodes[j]
                    if u != v:
                        if cnt[u] == 0:
                            touched[nt] = u; nt += 1
                        cnt[u] += 1; acc[u] += g
            if nt > 0:
                vals = np.empty(nt, np.int64)
                for i in range(nt):
                    vals[i] = cnt[touched[i]]
                o = np.argsort(-vals, kind='mergesort'); s = 0.0
                for i in range(min(b, nt)):
                    s += vals[o[i]]
                top_beta[v] = s
                sc = np.empty(nt)
                for i in range(nt):
                    sc[i] = acc[touched[i]]
                # only the b best partners per class can enter the inner maximum, so b per class are kept
                o = np.argsort(-sc, kind='mergesort'); fill = np.zeros(r, np.int64)
                for i in range(nt):
                    u = touched[o[i]]; c = cls[u]
                    if c >= 0 and fill[c] < b and sc[o[i]] > 0:
                        score[v, c, fill[c]] = sc[o[i]]; partner[v, c, fill[c]] = u; fill[c] += 1
                for i in range(nt):
                    cnt[touched[i]] = 0; acc[touched[i]] = 0.0
    return score, partner, top_beta


@nb.njit(cache=True)
def psi_min(g, me):
    """psi_E(g) = min_{t in 0..m_E} 1/(1+t) + g t and its minimiser t."""
    # [Paper] Proposition 7: psi_E
    t = 0
    while t < me and 1.0 / ((t + 1.0) * (t + 2.0)) > g:
        t += 1
    return 1.0 / (1.0 + t) + g * t, t


@nb.njit(parallel=True, cache=True)
def record_blocks(gam, v_ptr, v_sets, me, top_beta, n, out_k, out_c):
    """kappa_v = argmax_{kappa >= 0} sum_x psi(gamma_x + kappa) - kappa top_beta(v), by bisection on the
    supergradient, and the value of that block."""
    for v in nb.prange(n):
        if v_ptr[v + 1] == v_ptr[v]:
            out_k[v] = 0.0; out_c[v] = 0.0; continue
        lo = 0.0; hi = 0.5
        for it in range(40):
            mid = 0.5 * (lo + hi); st = 0.0
            for x in range(v_ptr[v], v_ptr[v + 1]):
                st += psi_min(gam[x] + mid, me[v_sets[x]])[1]
            if st - top_beta[v] > 0:
                lo = mid
            elif st - top_beta[v] < 0:
                hi = mid
            else:
                lo = mid; hi = mid
                break
        best = -1e300; bk = 0.0
        for k in (lo, hi, 0.0):
            val = -k * top_beta[v]
            for x in range(v_ptr[v], v_ptr[v + 1]):
                val += psi_min(gam[x] + k, me[v_sets[x]])[0]
            if val > best:
                best = val; bk = k
        out_k[v] = bk; out_c[v] = best


@nb.njit(cache=True)
def inner_max(mu, sv, b, take):
    """max_{|T| <= b} sum_{u in T} (w_vu - mu_c(u)) by greedy selection over the sorted per-class partner lists;
    take[i] is the number of partners taken from class i."""
    r = sv.shape[0]
    for i in range(r):
        take[i] = 0
    tot = 0.0
    for k in range(b):
        bi = -1; bv = 0.0
        for i in range(r):
            p = take[i]
            if p < b:
                w = sv[i, p]
                if w > 0:
                    val = w - mu[i]
                    if val > bv:
                        bv = val; bi = i
        if bi < 0:
            break
        tot += bv; take[bi] += 1
    return tot


@nb.njit(parallel=True, cache=True)
def inner_all(mu, score, b, cov, out_g, out_take):
    n = score.shape[0]
    for v in nb.prange(n):
        if cov[v]:
            out_g[v] = inner_max(mu, score[v], b, out_take[v])
        else:
            out_g[v] = 0.0
            for i in range(score.shape[1]):
                out_take[v, i] = 0


def ascend_prices(mu, score, cap, cov, iters, const, b):
    """Variable-target Polyak ascent on the class prices mu at fixed scores.

    Returns the best value, its mu and takes, and the value of the first evaluation."""
    n, r, _ = score.shape; g_v = np.zeros(n); take = np.zeros((n, r), np.int64); Dm = 1.0 / np.maximum(cap, 1.0)
    best = -np.inf; noimp = 0
    for it in range(iters):
        inner_all(mu, score, b, cov, g_v, take)
        L = const - g_v.sum() - float(np.dot(mu, cap))
        if it == 0:
            delta = 0.02 * abs(L); L0 = L
        if L > best:
            if best > -np.inf and L >= best + 0.5 * delta:
                delta *= 1.5
            best = L; bestmu = mu.copy(); besttake = take.copy(); noimp = 0
        else:
            noimp += 1
        # after 10 steps without improvement: halve the target gap and restart from the best prices
        if noimp >= 10:
            delta *= 0.5; noimp = 0; mu = bestmu.copy()
            if delta < 1e-7 * abs(best):
                break
            continue
        # projected supergradient: class demand minus capacity, zero where mu sits at its bound
        gr = take[cov].sum(axis=0).astype(np.float64) - cap
        gr[(mu <= 0) & (gr < 0)] = 0.0
        gg = float(np.dot(gr, Dm * gr))
        if gg <= 0:
            break
        mu = np.maximum(0.0, mu + (best + delta - L) / gg * Dm * gr)
    return best, bestmu, besttake, L0


@nb.njit(parallel=True, cache=True)
def pin_mate_counts(take, partner, v_ptr, v_sets, e_ptr, e_nodes, n, b, out_c):
    """out_c[x] = |T_v cap E| for every pin x = (v, E), with T_v the partners taken by the inner maximum."""
    r = partner.shape[1]
    for v in nb.prange(n):
        T = np.empty(b, np.int64); nt = 0
        for i in range(r):
            for j in range(take[v, i]):
                T[nt] = partner[v, i, j]; nt += 1
        for x in range(v_ptr[v], v_ptr[v + 1]):
            e = v_sets[x]; lo0 = e_ptr[e]; hi0 = e_ptr[e + 1]; c = 0
            for k in range(nt):
                # set members are sorted, so membership is a binary search
                u = T[k]; lo = lo0; hi = hi0
                while lo < hi:
                    mid = (lo + hi) // 2
                    if e_nodes[mid] < u:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < hi0 and e_nodes[lo] == u:
                    c += 1
            out_c[x] = c


@nb.njit(cache=True)
def class_term_f(s, d, a, b):
    best = a
    for kk in range(1, b + 1):
        val = (2.0 * kk * d - s) / (kk * (kk + 1.0))
        if val > best:
            best = val
    return best


@nb.njit(cache=True)
def class_term_phi(beta, d, a, S, b):
    """min over the breakpoints s in [0, S] of class_term_f(s) + beta s, and the minimising s."""
    best = class_term_f(0.0, d, a, b); s = 0.0
    val = class_term_f(S, d, a, b) + beta * S
    if val < best:
        best = val; s = S
    for j in range(1, b + 1):
        x = j * d
        if x < S:
            val = class_term_f(x, d, a, b) + beta * x
            if val < best:
                best = val; s = x
    for kk in range(1, b + 1):
        x = 2.0 * kk * d - a * kk * (kk + 1.0)
        if x > 0 and x < S:
            val = class_term_f(x, d, a, b) + beta * x
            if val < best:
                best = val; s = x
    return best, s


@nb.njit(cache=True)
def inner_max_scaled(beta, mu, sv, b, take):
    """Greedy inner maximum of sum_{u in T} (beta w_vu - mu_c(u)); also returns sum_{u in T} w_vu."""
    r = sv.shape[0]
    for i in range(r):
        take[i] = 0
    tot = 0.0; gw = 0.0
    for k in range(b):
        bi = -1; bv = 0.0
        for i in range(r):
            p = take[i]
            if p < b:
                w = sv[i, p]
                if w > 0:
                    val = beta * w - mu[i]
                    if val > bv:
                        bv = val; bi = i
        if bi < 0:
            break
        tot += bv; gw += sv[bi, take[bi]]; take[bi] += 1
    return tot, gw


@nb.njit(parallel=True, cache=True)
def class_bound_terms(mu, d, a, smax, score, b, out_term, out_beta, out_take):
    """Per-record term of the class-capacity bound: beta_v by bisection on s*(beta) - sum_T w, then the term value."""
    n, r, _ = score.shape
    for v in nb.prange(n):
        take = out_take[v]
        if d[v] <= 0:
            out_term[v] = 0.0; out_beta[v] = 0.0
            for i in range(r):
                take[i] = 0
            continue
        lo = 0.0; hi = 0.5
        for it in range(30):
            mid = 0.5 * (lo + hi)
            ph, s = class_term_phi(mid, d[v], a[v], smax[v], b)
            tot, gw = inner_max_scaled(mid, mu, score[v], b, take)
            if s - gw > 0:
                lo = mid
            elif s - gw < 0:
                hi = mid
            else:
                lo = mid; hi = mid
                break
        bestv = -1e300; bestb = 0.0
        for cand in (lo, hi, 0.0):
            ph, s = class_term_phi(cand, d[v], a[v], smax[v], b)
            tot, gw = inner_max_scaled(cand, mu, score[v], b, take)
            if ph - tot > bestv:
                bestv = ph - tot; bestb = cand
        ph, s = class_term_phi(bestb, d[v], a[v], smax[v], b)
        tot, gw = inner_max_scaled(bestb, mu, score[v], b, take)
        out_term[v] = ph - tot; out_beta[v] = bestb


def class_bound(score, cap, iters, d, a, top_beta, cov, b):
    """Variable-target Polyak ascent on mu for the class-capacity bound. Returns the best value, mu and the per-record
    beta_v, which seeds gamma."""
    n, r, _ = score.shape; term = np.zeros(n); beta = np.zeros(n); take = np.zeros((n, r), np.int64)
    Dm = 1.0 / np.maximum(cap, 1.0)
    mu = np.zeros(r); best = -np.inf; noimp = 0
    for it in range(iters):
        class_bound_terms(mu, d, a, top_beta, score, b, term, beta, take)
        L = float(term[cov].sum() - np.dot(mu, cap))
        if it == 0:
            delta = 0.05 * abs(L)
        if L > best:
            if best > -np.inf and L >= best + 0.5 * delta:
                delta *= 1.5
            best = L; bestmu = mu.copy(); bestbeta = beta.copy(); noimp = 0
        else:
            noimp += 1
        if noimp >= 10:
            delta *= 0.5; noimp = 0; mu = bestmu.copy()
            if delta < 1e-6 * abs(best):
                break
            continue
        g = take[cov].sum(axis=0).astype(np.float64) - cap
        g[(mu <= 0) & (g < 0)] = 0.0
        gg = float(np.dot(g, Dm * g))
        if gg <= 0:
            break
        mu = np.maximum(0.0, mu + (best + delta - L) / gg * Dm * g)
    return best, bestmu, bestbeta


@nb.njit(parallel=True, cache=True)
def project_to_average(gam, kap, cbar, v_ptr, v_sets, me, eta, n):
    """Move gamma_x + kappa_v toward the subdifferential of the piecewise-linear interpolation of h at the averaged
    count cbar_x: the slope 1/((c+1)(c+2)) of the active piece when cbar_x is fractional, the interval
    [1/((c+1)(c+2)), 1/(c(c+1))] when it is an integer c. gamma stays in [0, 1/2]."""
    for v in nb.prange(n):
        for x in range(v_ptr[v], v_ptr[v + 1]):
            cb = cbar[x]; m = me[v_sets[x]]; c = int(np.floor(cb + 1e-9))
            if cb - c > 1e-9:
                lo = 1.0 / ((c + 1.0) * (c + 2.0)); hi = lo
            else:
                lo = 1.0 / ((c + 1.0) * (c + 2.0)) if c < m else 0.0
                hi = 1.0 / (c * (c + 1.0)) if c > 0 else 0.5
            gk = gam[x] + kap[v]; tgt = min(max(gk, lo), hi)
            gam[x] = min(0.5, max(0.0, gam[x] + eta * (tgt - gk)))


def degree_classes(d, G):
    """Covered records by degree (stable); classes of width 1, 2, 4, ...; the last of G classes takes the rest."""
    cls = np.full(len(d), -1, np.int64); cov_ids = np.flatnonzero(d > 0)
    rank = cov_ids[np.argsort(-d[cov_ids], kind='stable')]; start = 0; c = 0; width = 1
    while start < len(rank):
        end = len(rank) if c == G - 1 else min(len(rank), start + width)
        cls[rank[start:end]] = c; start = end; c += 1; width *= 2
    return cls, c


@nb.njit(cache=True)
def in_loads(take, partner, n, out_q):
    """out_q[u] = #{v : u in T_v}, the number of records whose inner maximum takes u."""
    r = partner.shape[1]
    for u in range(n):
        out_q[u] = 0.0
    for v in range(n):
        for i in range(r):
            for j in range(take[v, i]):
                out_q[partner[v, i, j]] += 1.0


def load_bins(qbar, b):
    """Bucket of the averaged in-load: qbar <= beta, <= 2 beta, <= 8 beta, above."""
    return np.where(qbar <= b, 0, np.where(qbar <= 2 * b, 1, np.where(qbar <= 8 * b, 2, 3)))


def split_classes(cls, qbar, b):
    """Split every class by load_bins into at most 4 children (empty children dropped).

    Returns the new class of every record and the parent of every child."""
    binq = load_bins(qbar, b)
    has = cls >= 0; key = cls[has] * 4 + binq[has]
    uq, inv = np.unique(key, return_inverse=True)
    new = np.full(len(cls), -1, np.int64); new[has] = inv
    return new, (uq // 4).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='.npz with e_ptr, e_nodes (for example the test group unions)')
    ap.add_argument('--out', required=True, help='output JSON (rewritten after every outer iteration)')
    ap.add_argument('--save', default=None, help='optional .npz for the best point (gamma, kappa, mu, classes)')
    ap.add_argument('--B', type=int, default=12, help='page capacity (records per page)')
    ap.add_argument('--records', type=int, default=0,
                    help='number of records n of the index; default: largest identifier + 1')
    ap.add_argument('--classes', type=int, default=20, help='number of initial degree classes')
    ap.add_argument('--class-iters', type=int, default=400, help='iterations of the class stage')
    ap.add_argument('--outer-a', type=int, default=20, help='outer iterations before the class split (stage A)')
    ap.add_argument('--outer-b', type=int, default=15, help='outer iterations after the class split (stage B)')
    ap.add_argument('--price-iters', type=int, default=60, help='price-ascent iterations per outer iteration')
    ap.add_argument('--load-from', type=int, default=10,
                    help='stage-A outer iterations from this index on are averaged for the in-loads of the split')
    A = ap.parse_args()
    B = A.B; b = B - 1
    t0 = time.time()
    z = np.load(A.sets); e_ptr = z['e_ptr'].astype(np.int64); e_nodes = z['e_nodes'].astype(np.int32).copy()
    m = len(e_ptr) - 1; n = max(int(e_nodes.max()) + 1, A.records)
    sort_set_members(e_ptr, e_nodes)
    # incidence lists: pins x of record v are v_ptr[v]..v_ptr[v+1], in ascending set order
    sz = np.diff(e_ptr); d = np.bincount(e_nodes, minlength=n).astype(np.float64)
    me = np.minimum(b, sz - 1).astype(np.int64)
    _, _, v_ptr, v_sets, _ = incidence(e_ptr, e_nodes, n)
    pin_v = np.repeat(np.arange(n), np.diff(v_ptr))
    cov = d > 0
    cls, r = degree_classes(d, A.classes)
    assert (cls[cov] >= 0).all(), 'a covered record has no class'
    csize = np.bincount(cls[cls >= 0], minlength=r).astype(np.int64); cap = b * csize
    R = dict(sets=m, n=n, covered=int(cov.sum()), pins=int(v_ptr[-1]), B=B, classes=r, class_iters=A.class_iters,
             outer_a=A.outer_a, outer_b=A.outer_b, price_iters=A.price_iters, load_from=A.load_from, history=[])
    # a_v = sum_E 1 / min(|E|, B): per-record constant of the class-capacity bound
    a = np.bincount(e_nodes, weights=np.repeat(1.0 / np.minimum(sz, B), sz), minlength=n)
    gam = np.zeros(int(v_ptr[-1])); kap = np.zeros(n); cst = np.zeros(n); best = -np.inf
    cnt = np.zeros(len(gam), np.int64)
    t = time.time()
    score, partner, top_beta = class_partner_scores(e_ptr, e_nodes, v_ptr, v_sets, n, cls, r, b, np.ones(len(gam)),
                                                    nb.get_num_threads())
    L1, mu, beta1 = class_bound(score, cap, A.class_iters, d, a, top_beta, cov, b)
    R['class_bound_per_set'] = L1 / m
    gam[:] = beta1[pin_v]
    log('class stage bound/m', round(L1 / m, 4), 'classes', r, f'{time.time() - t:.0f}s')
    cbar = np.zeros(len(gam)); qsum = np.zeros(n); qn = 0; q = np.zeros(n)
    st = None
    for k in range(A.outer_a + A.outer_b):
        stage = 'A' if k < A.outer_a else 'B'
        if k == A.outer_a:
            # split the classes once, by the in-loads averaged over stage A; restart from the best stage-A point
            qbar = qsum / qn
            cls_new, parent = split_classes(cls, qbar, b)
            assert (cls_new[cov] >= 0).all(), 'a covered record has no class after the split'
            mu = st['mu'][parent]; cls, r = cls_new, len(parent)
            csize = np.bincount(cls[cls >= 0], minlength=r).astype(np.int64); cap = b * csize; gam[:] = st['gam']
            R['split'] = dict(classes=int(r), bins=np.bincount(load_bins(qbar[cov], b), minlength=4).tolist(),
                              from_iteration=int(st['k']))
            log('split classes', r, 'bins', R['split']['bins'])
            Lpar = st['L']
        t = time.time()
        score, partner, top_beta = class_partner_scores(e_ptr, e_nodes, v_ptr, v_sets, n, cls, r, b, gam,
                                                        nb.get_num_threads())
        record_blocks(gam, v_ptr, v_sets, me, top_beta, n, kap, cst)
        L, mu, take, L0 = ascend_prices(mu, score, cap, cov, A.price_iters, float(cst[cov].sum()), b)
        if k == A.outer_a:
            # children inherit their parent's price, so the first value on the refined classes equals the parent's
            rel = abs(L0 - Lpar) / abs(Lpar)
            R['split'].update(parent_value=Lpar / m, first_value=L0 / m, rel=rel)
            log('split check parent/m', Lpar / m, 'refined first/m', L0 / m, 'rel', rel)
            if rel > 1e-9:
                R['status'] = 'split check failed'; json.dump(R, open(A.out, 'w'), indent=1)
                log('split check failed'); sys.exit(2)
        if L > best:
            best = L; st = dict(gam=gam.copy(), kap=kap.copy(), mu=mu.copy(), cls=cls.copy(), L=L, k=k)
        if A.load_from <= k < A.outer_a:
            in_loads(take, partner, n, q); qsum += q; qn += 1
        pin_mate_counts(take, partner, v_ptr, v_sets, e_ptr, e_nodes, n, b, cnt)
        R['history'].append([k, stage, r, L / m, best / m, time.time() - t])
        log('outer', k, stage, 'classes', r, 'L/m', round(L / m, 4), 'best/m', round(best / m, 4),
            f'{time.time() - t:.0f}s')
        R['L_best_per_set'] = best / m; json.dump(R, open(A.out, 'w'), indent=1)
        # running mean of the page-mate counts, then the gamma step toward its subdifferential
        cbar += (cnt - cbar) / (k + 1.0)
        project_to_average(gam, kap, cbar, v_ptr, v_sets, me, 0.5, n)
    if A.save:
        np.savez(A.save, gam=st['gam'], kap=st['kap'], mu=st['mu'], cls=st['cls'], L=st['L'], k=st['k'])
    R['secs'] = time.time() - t0; json.dump(R, open(A.out, 'w'), indent=1)
    log('done: lower bound per set', R['L_best_per_set'])


if __name__ == '__main__':
    main()
