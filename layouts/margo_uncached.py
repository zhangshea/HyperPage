"""MARGO on uncached records: MARGO's partitioner on the graph induced by the records a budget leaves uncached, with the
cached records filling the free slots in MARGO's page order. Runs the user's own build of MARGO's partitioner, which
is not shipped; this script only writes its input files and reads its output file."""
# [Paper] Section VII-A, Compared methods, "MARGO's partitioner on the uncached records"; Section VII-C and Table VII,
# row "MARGO on uncached records"; Table IV, row "21M, 25%".
import argparse
import json
import os
import subprocess
import time

import numpy as np
import numba as nb

from hyperpage.hypergraph import most_frequent_records


@nb.njit(cache=True)
def parse_lists(raw, pos0, n):
    """n length-prefixed uint32 lists starting at word pos0 of raw, as CSR (ptr, values)."""
    ptr = np.zeros(n + 1, np.int64); pos = pos0
    for i in range(n):
        d = raw[pos]; ptr[i + 1] = ptr[i] + d; pos += 1 + d
    val = np.empty(ptr[n], np.uint32); pos = pos0
    for i in range(n):
        d = raw[pos]; pos += 1
        for j in range(d):
            val[ptr[i] + j] = raw[pos + j]
        pos += d
    return ptr, val


@nb.njit(cache=True)
def induce(ptr, nbr, wbits, keep, newid):
    """Graph and arc weights induced by the kept records, renumbered by newid, as length-prefixed lists.

    Returns (lists of neighbours, lists of weights, kept arcs, kept records without a kept neighbour)."""
    n = keep.shape[0]; nk = 0; ne = 0
    for i in range(n):
        if keep[i]:
            nk += 1
            for j in range(ptr[i], ptr[i + 1]):
                if keep[nbr[j]]:
                    ne += 1
    oi = np.empty(nk + ne, np.uint32); ow = np.empty(nk + ne, np.uint32); p = 0; iso = 0
    for i in range(n):
        if keep[i]:
            c = 0
            for j in range(ptr[i], ptr[i + 1]):
                if keep[nbr[j]]:
                    c += 1
            oi[p] = c; ow[p] = c; p += 1
            if c == 0:
                iso += 1
            for j in range(ptr[i], ptr[i + 1]):
                v = nbr[j]
                if keep[v]:
                    oi[p] = newid[v]; ow[p] = wbits[j]; p += 1
    return oi, ow, ne, iso


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index-prefix', required=True,
                    help="prefix of MARGO's index files: <prefix>_mem.index, <prefix>_mem.index.weights and "
                         "<prefix>_disk.index (header only)")
    ap.add_argument('--vectors', required=True,
                    help='record vectors: int32 n, int32 d, then n * d values of --vector-type')
    ap.add_argument('--vector-type', required=True, choices=('int8', 'float32'), help='element type of --vectors')
    ap.add_argument('--sets', required=True, help='training group unions of one draw (.npz with e_ptr, e_nodes)')
    ap.add_argument('--margo-layout', required=True,
                    help="MARGO's layout of the full index (.npy); its page order places the cached records")
    ap.add_argument('--partitioner', required=True,
                    help="path of the user's build of MARGO's graph partitioner (new_mincut)")
    ap.add_argument('--cached-pages', type=int, required=True,
                    help='page slots of the record cache, floor(rho * D); it holds capacity times as many records')
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--work-dir', required=True, help="directory for the partitioner's input and output files")
    ap.add_argument('--keep-inputs', action='store_true', help="keep the induced index and vector files after the run")
    ap.add_argument('--out', required=True, help='output layout (.npy); run details go to the same name with .json')
    a = ap.parse_args()

    B = a.capacity
    os.makedirs(a.work_dir, exist_ok=True); W = a.work_dir; t0 = time.time()
    # graph and MARGO's arc weights, aligned list by list
    graph_file = a.index_prefix + '_mem.index'; weight_file = a.index_prefix + '_mem.index.weights'
    raw = np.fromfile(graph_file, dtype=np.uint32)
    with open(graph_file, 'rb') as f:
        np.fromfile(f, np.uint64, 1)
        maxdeg, entry = np.fromfile(f, np.uint32, 2)
        frozen, = np.fromfile(f, np.uint64, 1)
    with open(a.index_prefix + '_disk.index', 'rb') as f:
        nr, nc = np.fromfile(f, np.int32, 2); meta = np.fromfile(f, np.uint64, nr)
    n = int(meta[0]); ptr, nbr = parse_lists(raw, 6, n); del raw
    wraw = np.fromfile(weight_file, dtype=np.uint32)
    with open(weight_file, 'rb') as f:
        maxw, = np.fromfile(f, np.float32, 1); ne_all, npts_w = np.fromfile(f, np.uint64, 2)
    assert int(npts_w) == n, (npts_w, n)
    wptr, wbits = parse_lists(wraw, 5, n); del wraw
    assert np.array_equal(ptr, wptr), 'weights not aligned with the graph'; del wptr

    # [Paper] Definition 7: the record cache holds the B * floor(rho * D) most frequent training records,
    # ties by identifier
    keep = np.ones(n, np.bool_)
    if a.cached_pages > 0:
        z = np.load(a.sets)
        keep[most_frequent_records(z['e_ptr'], z['e_nodes'].astype(np.int64), n, B * a.cached_pages)] = False
        del z
    kept = np.nonzero(keep)[0]; nk = len(kept); newid = np.full(n, -1, np.int64); newid[kept] = np.arange(nk)
    oi, ow, ne, iso = induce(ptr, nbr, wbits, keep, newid); del ptr, nbr, wbits
    # the induced index: graph file, weight file and a disk-index header with nk records and B records per page
    st = int(newid[entry]) if keep[entry] else 0
    with open(f'{W}/m_mem.index', 'wb') as f:
        np.array([24 + 4 * len(oi)], np.uint64).tofile(f); np.array([maxdeg, st], np.uint32).tofile(f)
        np.array([frozen], np.uint64).tofile(f); oi.tofile(f)
    with open(f'{W}/m_mem.index.weights', 'wb') as f:
        np.array([maxw], np.float32).tofile(f); np.array([ne, nk], np.uint64).tofile(f); ow.tofile(f)
    del oi, ow
    m2 = meta.copy(); m2[0] = nk; m2[4] = B
    with open(f'{W}/m_disk.index', 'wb') as f:
        np.array([nr, nc], np.int32).tofile(f); m2.tofile(f)
    # float32 vectors of the kept records, in new identifier order
    with open(a.vectors, 'rb') as f:
        bn, d = np.fromfile(f, np.int32, 2)
    assert bn == n, (bn, n)
    X = np.memmap(a.vectors, dtype=np.int8 if a.vector_type == 'int8' else np.float32, mode='r', offset=8, shape=(n, d))
    chunk = 1 << 20
    with open(f'{W}/base.fbin', 'wb') as f:
        np.array([nk, d], np.int32).tofile(f)
        for c in range(0, n, chunk):
            X[c:c + chunk][keep[c:c + chunk]].astype(np.float32).tofile(f)
    del X
    prep_seconds = time.time() - t0

    # MARGO's partitioner, run unchanged on the induced index
    t1 = time.time()
    r = subprocess.run([a.partitioner, f'{W}/m', f'{W}/base.fbin'], capture_output=True, text=True)
    cut_seconds = time.time() - t1
    open(f'{W}/partitioner.log', 'w').write(r.stdout + '\n' + r.stderr)
    assert r.returncode == 0, r.stderr[-2000:]
    with open(f'{W}/m_partition.bin', 'rb') as f:
        _, npg, npts = np.fromfile(f, dtype=np.uint64, count=3).astype(np.int64)
        sizes = np.zeros(npg, np.int64)
        for p in range(npg):
            sz = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
            np.fromfile(f, dtype=np.uint32, count=sz)
            sizes[p] = sz
        id2p = np.fromfile(f, dtype=np.uint32, count=nk).astype(np.int64)
    assert npts == nk and npg == -(-nk // B)
    assert np.array_equal(np.bincount(id2p, minlength=npg), sizes) and sizes.max() <= B

    # cached records: free slots of partial pages first, then new pages, in MARGO's page order with ties by identifier
    page = np.full(n, -1, np.int64); page[kept] = id2p
    cached = np.nonzero(~keep)[0]; mg = np.load(a.margo_layout).astype(np.int64)
    cached = cached[np.lexsort((cached, mg[cached]))]; k = 0
    for p in np.nonzero(sizes < B)[0]:
        take = min(B - int(sizes[p]), len(cached) - k)
        page[cached[k:k + take]] = p; sizes[p] += take; k += take
    nxt = int(npg)
    while k < len(cached):
        page[cached[k:k + B]] = nxt; k += B; nxt += 1
    cnt = np.bincount(page, minlength=nxt)
    assert (page >= 0).all() and cnt.max() <= B and nxt == -(-n // B), (nxt, -(-n // B), cnt.max())
    np.save(a.out, page.astype(np.int32))
    info = dict(layout='MARGO on uncached records', cached_pages=a.cached_pages, capacity=B, records=int(n),
                kept=int(nk),
                cached=int(n - nk), arcs_kept=int(ne), arcs_all=int(ne_all), isolated_kept=int(iso), pages=int(nxt),
                pages_partitioned=int(npg), pages_not_full=int((cnt != B).sum()), prep_seconds=prep_seconds,
                partitioner_seconds=cut_seconds, seconds=time.time() - t0)
    json.dump(info, open(a.out[:-4] + '.json' if a.out.endswith('.npy') else a.out + '.json', 'w'), indent=1)
    print(json.dumps(info))
    if not a.keep_inputs:
        for fn in ('base.fbin', 'm_mem.index', 'm_mem.index.weights'):
            os.remove(f'{W}/{fn}')


if __name__ == '__main__':
    main()
