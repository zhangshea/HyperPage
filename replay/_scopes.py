"""Page reads under read-once scopes: every scope reads each page of its records once, with no state across scopes.
Helper module of the scripts in replay/, imported by name: Python puts a script's directory on the import path."""
import numpy as np
import numba as nb


@nb.njit(cache=True)
def scope_reads(order, e_ptr, e_nodes, scope_of_set, n_scopes, page, npages):
    """Distinct pages per scope. Sets arrive in `order`; the sets of one scope must arrive together."""
    # [Paper] Corollary 1: a read-once scope S_sigma reads exactly lambda_pi(E_sigma) pages
    stamp = np.full(npages, -1, np.int64); res = np.zeros(n_scopes, np.int64)
    for c in order:
        g = scope_of_set[c]
        for j in range(e_ptr[c], e_ptr[c + 1]):
            p = page[e_nodes[j]]
            if stamp[p] != g:
                stamp[p] = g; res[g] += 1
    return res


def group_reads(u_ptr, u_nodes, page, npages):
    """lambda_pi(E_b) of every group union, in the order of the unions file."""
    every = np.arange(len(u_ptr) - 1, dtype=np.int64)
    return scope_reads(every, u_ptr, u_nodes, every, len(every), page, npages)


def tool_call_scopes(schedule):
    """Tool-call scope of every sub-query and the turn-lockstep order in which the scopes arrive.

    A tool call is one request of one rollout at one turn of one step, the unique (step, rollout, turn)."""
    s = schedule
    order = np.lexsort((s['position'], s['rollout'], s['turn'], s['step'])).astype(np.int64)
    _, call = np.unique(np.stack([s['step'], s['rollout'], s['turn']]), axis=1, return_inverse=True)
    call = call.ravel().astype(np.int64)
    return order, call, int(call.max()) + 1


def expand(spec, draws):
    """NAME=FILE with an optional {draw} in FILE -> (name, [file per draw]); a FILE without {draw} is one layout."""
    name, pattern = spec.split('=', 1)
    return name, ([pattern.format(draw=k) for k in draws] if '{draw}' in pattern else [pattern])
