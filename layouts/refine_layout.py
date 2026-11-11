"""Build one layout with Algorithm 1: refine an incumbent layout under one objective, cache-blind or cache-aware."""
# [Paper] Algorithm 1 and Section VI. The layout labels of the paper map to arguments as follows (G: group unions,
# Q: per-query sets, C: residual unions of one budget, CF: C with the records outside the training unions frozen):
#   ARC-G, CLQ-G, SET-G, P05-G   --objective ARC|CLQ|SET|P05 --sets <training group unions>
#   ARC-Q, SET-Q                 --objective ARC|SET --sets <training per-query sets>
#   ARC-C, P05-C                 the G command plus --cached-pages floor(rho * D)
#   ARC-CF                       the ARC-C command plus --freeze-uncovered
#   HEAT (on residual unions)    --objective HEAT --query-sets <training per-query sets> [--cached-pages ...]
#   STARLING                     --objective STARLING --start <identifier-order layout>
import argparse
import json

import numpy as np

from hyperpage.formats import read_index_meta, parse_graph
from hyperpage.layout import OBJECTIVES, refine_layout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index-prefix', required=True,
                    help='prefix of the index files (<prefix>_disk.index, <prefix>_mem.index)')
    ap.add_argument('--sets', default=None, help='training hypergraph (.npz with e_ptr, e_nodes); not used by STARLING')
    ap.add_argument('--start', required=True, help='incumbent layout (.npy, page of every record)')
    ap.add_argument('--objective', required=True, choices=OBJECTIVES)
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--seed', type=int, default=0,
                    help='seed of the visiting order; the paper uses the index of the training draw (0, 1, 2)')
    ap.add_argument('--max-passes', type=int, default=8)
    ap.add_argument('--tol', type=float, default=1e-4, help='stop when a pass gains at most tol times the objective')
    ap.add_argument('--max-cand', type=int, default=16, help='candidate pages per record')
    ap.add_argument('--max-sets', type=int, default=64, help='sets per record scanned for candidate pages')
    ap.add_argument('--window', type=int, default=32,
                    help='positions around the record inside a set scanned for candidate pages')
    ap.add_argument('--cached-pages', type=int, default=0,
                    help='cache-aware refinement: page slots of the record cache, floor(rho * D); '
                         'it holds capacity times as many records')
    ap.add_argument('--cached-records', type=int, default=None,
                    help='exact number of cached records instead of capacity * cached-pages')
    ap.add_argument('--freeze-uncovered', action='store_true',
                    help='keep records outside the training sets on their pages')
    ap.add_argument('--query-sets', default=None, help='training per-query sets (.npz), needed by HEAT')
    ap.add_argument('--heat-without-arc-term', action='store_true',
                    help='HEAT weights from traversal counts alone')
    ap.add_argument('--out', required=True, help='output layout (.npy); run details go to the same name with .json')
    a = ap.parse_args()

    n = read_index_meta(a.index_prefix + '_disk.index')['npts']
    graph = parse_graph(np.fromfile(a.index_prefix + '_mem.index', dtype=np.uint32), n)
    if a.objective == 'STARLING':
        sets = (np.zeros(1, np.int64), np.zeros(0, np.int32))
    else:
        assert a.sets is not None, '--sets is required for this objective'
        z = np.load(a.sets)
        sets = (z['e_ptr'], z['e_nodes'])
    query_sets = None
    if a.objective == 'HEAT':
        q = np.load(a.query_sets)
        query_sets = (q['e_ptr'], q['e_nodes'])
    cached = a.cached_records if a.cached_records is not None else a.capacity * a.cached_pages
    start = np.load(a.start)
    assert len(start) == n, 'the start layout does not match the index'
    page, info = refine_layout(a.objective, sets, graph, start, a.capacity, a.seed, a.max_passes, a.tol, a.max_cand,
                               a.max_sets, a.window, cached, a.freeze_uncovered, query_sets,
                               not a.heat_without_arc_term)
    np.save(a.out, page.astype(np.int32))
    info_path = a.out[:-4] + '.json' if a.out.endswith('.npy') else a.out + '.json'
    json.dump(info, open(info_path, 'w'), indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
