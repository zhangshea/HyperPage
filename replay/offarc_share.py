"""Off-arc share of co-located pairs: the share of record pairs on one page inside a group that no graph arc joins."""
# [Paper] Section VII-C: dropping the arc restriction raises the off-arc share of co-located pairs on the W3-1M test
# groups from 0.404 (ARC-G) to 0.787 (CLQ-G), draw 0.
import argparse
import json

import numpy as np
import numba as nb

from hyperpage import formats


@nb.njit(cache=True)
def colocated_pairs(e_ptr, e_nodes, page, npages, g_ptr, g_nbr, n):
    """Co-located pairs and co-located arc pairs summed over the sets, each set counted unweighted.

    A pair is co-located if both records are in the set and on one page; an arc pair is a co-located pair joined by an
    arc in either direction, counted once."""
    cnt = np.zeros(npages, np.int64); mark = np.full(n, -1, np.int64)
    pairs = 0; arcs = 0
    for e in range(len(e_ptr) - 1):
        for j in range(e_ptr[e], e_ptr[e + 1]):
            cnt[page[e_nodes[j]]] += 1; mark[e_nodes[j]] = e
        for j in range(e_ptr[e], e_ptr[e + 1]):
            p = page[e_nodes[j]]
            if cnt[p] > 0:
                pairs += cnt[p] * (cnt[p] - 1) // 2; cnt[p] = 0
        for j in range(e_ptr[e], e_ptr[e + 1]):
            u = e_nodes[j]
            for t in range(g_ptr[u], g_ptr[u + 1]):
                v = g_nbr[t]
                if mark[v] == e and page[v] == page[u] and v != u:
                    # count the unordered pair once: from the smaller record, or from u if the reverse arc is absent
                    rev = False
                    for t2 in range(g_ptr[v], g_ptr[v + 1]):
                        if g_nbr[t2] == u:
                            rev = True; break
                    if u < v or not rev:
                        arcs += 1
    return pairs, arcs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--unions', required=True, help='group unions of the test split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--graph', required=True, help='the in-memory Vamana graph file of the index (_mem.index)')
    ap.add_argument('--records', type=int, required=True, help='records of the index')
    ap.add_argument('--pages', type=int, required=True, help='pages of the index')
    ap.add_argument('--layout', action='append', required=True, metavar='NAME=FILE', help='one layout file per name')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    u = np.load(a.unions)
    g_ptr, g_nbr = formats.parse_graph(np.fromfile(a.graph, dtype=np.uint32), a.records)
    out = {}
    for spec in a.layout:
        name, f = spec.split('=', 1)
        pairs, arcs = colocated_pairs(u['e_ptr'].astype(np.int64), u['e_nodes'].astype(np.int32),
                                      np.load(f).astype(np.int32), a.pages, g_ptr, g_nbr, a.records)
        # off-arc share = 1 - (co-located arc pairs) / (co-located pairs); arcs are those of Definition 9's ARC weights
        out[name] = dict(colocated_pairs=int(pairs), colocated_arc_pairs=int(arcs),
                         off_arc_share=float(1 - arcs / pairs) if pairs else None)
        share = out[name]['off_arc_share']
        print(f'{name:14s} co-located pairs {pairs:,}, on arcs {arcs:,}, '
              f'off-arc share {"undefined" if share is None else format(share, ".3f")}')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
