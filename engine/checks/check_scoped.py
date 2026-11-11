"""Check one run of the read-once scoped engine against the stock reference run (check i) and against the replayed
reads of every scope (check ii)."""
# (i)  Assumption 1: result ids, distances, per-call expanded-node sequences and round boundaries are identical to the
#      stock engine on the stock layout, call by call.
# (ii) the engine's page reads per scope equal lambda_pi(E_sigma), scope by scope; the pages read in each scope are
#      distinct and are exactly the pages of the union of its expanded nodes; reads + hits + shared reads = expanded.
# [Paper] Section VII-A: "A patched DiskANN reproduces the stock results and expansion sequences of 7,850 W3
# sub-queries", and Corollary 1 for the reads of read-once scopes. Prints one JSON line.
import argparse
import json
import os

import numpy as np

from hyperpage import formats


def read_res_full(prefix):
    """Result ids and distances of a run (<prefix>.res.bin)."""
    with open(prefix + '.res.bin', 'rb') as f:
        nq, k = np.fromfile(f, dtype=np.uint32, count=2)
        ids = np.fromfile(f, dtype=np.uint32, count=nq * k).reshape(nq, k)
        d = np.fromfile(f, dtype=np.float32, count=nq * k).reshape(nq, k)
    return ids, d


def read_scopes(prefix):
    """Per-scope counters of batch_search (<prefix>.batches.bin, 13 columns)."""
    with open(prefix + '.batches.bin', 'rb') as f:
        nb = int(np.fromfile(f, np.uint64, 1)[0]); rows = np.fromfile(f, np.uint64)
    assert len(rows) == nb * 13
    rows = rows.reshape(nb, 13)
    return {k: rows[:, i].astype(np.int64) for i, k in
            enumerate(['reads', 'hits', 'dedup', 'calls', 'expanded', 'lat_ns', 'thread', 'cpu', 'io_ns', 'pq_ns',
                       'nbr_ns', 'rounds', 'rounds_noread'])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference',
                    help='prefix of the stock run of search_and_log on the stock index (.log.bin, .res.bin)')
    ap.add_argument('run', help='prefix of the batch_search run in mode batch with log 1')
    ap.add_argument('scope_ptr', help='scope pointers (u64) used by the run')
    ap.add_argument('layout', help='layout of the run (.npy)')
    ap.add_argument('lambda_', metavar='lambda', help='replayed reads per scope (.npy from stage_scoped_sample.py)')
    ap.add_argument('--proc-read-bytes', type=int, default=None,
                    help='proc_read_bytes of the run summary line: then also check bytes = reads x page size')
    ap.add_argument('--page-size', type=int, default=8192)
    a = ap.parse_args()

    bptr = np.fromfile(a.scope_ptr, np.uint64).astype(np.int64)
    page = np.load(a.layout).astype(np.int64); lam = np.load(a.lambda_).astype(np.int64)
    (i1, d1), (i2, d2) = read_res_full(a.reference), read_res_full(a.run)
    A, R = formats.read_log(a.reference), formats.read_log(a.run)
    Bt = read_scopes(a.run)
    nb = len(bptr) - 1
    out = dict(run=os.path.basename(a.run), layout=os.path.basename(a.layout), calls=int(A['nq']), scopes=int(nb))
    # (i) Assumption 1
    out['ids_identical'] = bool(np.array_equal(i1, i2))
    out['dists_identical'] = bool(np.array_equal(d1, d2))
    out['expanded_sequences_identical'] = bool(np.array_equal(A['noff'], R['noff'])
                                               and np.array_equal(A['nodes'], R['nodes']))
    out['rounds_identical'] = bool(np.array_equal(A['roff'], R['roff'])
                                   and np.array_equal(A['round_end'], R['round_end']))
    out['parents_identical'] = bool(np.array_equal(A['parents'], R['parents']))
    out['check_i'] = all(out[k] for k in ('ids_identical', 'dists_identical', 'expanded_sequences_identical',
                                          'rounds_identical'))
    # (ii) reads of every scope
    assert len(lam) == nb and len(Bt['reads']) == nb
    eng = Bt['reads']
    out['reads_equal_lambda_all_scopes'] = bool(np.array_equal(eng, lam))
    out['scopes_mismatch'] = int((eng != lam).sum())
    out['engine_reads_total'] = int(eng.sum()); out['lambda_total'] = int(lam.sum())
    out['stock_reads_total'] = int(len(A['sectors']))   # one read per expanded node in the stock engine
    # issued pages per scope, from the run's log (read_sectors holds issued reads only)
    cb = np.repeat(np.arange(nb), np.diff(bptr))                     # scope of each call
    sec_b = np.repeat(cb, np.diff(R['soff'])); secs = R['sectors']
    node_b = np.repeat(cb, np.diff(R['noff'])); nodes = R['nodes'].astype(np.int64)
    P = int(page.max()) + 2
    issued = sec_b * P + secs
    want = np.unique(node_b * P + 1 + page[nodes])
    out['issued_pages_distinct_per_scope'] = bool(len(np.unique(issued)) == len(issued))
    out['issued_pages_equal_union_pages'] = bool(np.array_equal(np.sort(issued), want))
    out['log_reads_equal_counter'] = bool(len(secs) == eng.sum())
    out['reads_hits_shared_sum_to_expanded'] = bool(np.array_equal(eng + Bt['hits'] + Bt['dedup'], Bt['expanded']))
    out['expanded_equal_stock_expanded'] = bool(np.array_equal(Bt['expanded'],
                                                               np.add.reduceat(np.diff(A['noff']), bptr[:-1])))
    out['calls_per_scope_ok'] = bool(np.array_equal(Bt['calls'], np.diff(bptr)))
    out['hits_total'] = int(Bt['hits'].sum()); out['shared_reads_total'] = int(Bt['dedup'].sum())
    if a.proc_read_bytes is not None:
        out['proc_read_bytes_equal_reads_x_page'] = bool(a.proc_read_bytes == eng.sum() * a.page_size)
    out['check_ii'] = all(out[k] for k in ('reads_equal_lambda_all_scopes', 'issued_pages_distinct_per_scope',
                                           'issued_pages_equal_union_pages', 'log_reads_equal_counter',
                                           'reads_hits_shared_sum_to_expanded', 'calls_per_scope_ok'))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
