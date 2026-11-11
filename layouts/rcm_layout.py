"""RCM baseline: reverse Cuthill-McKee order of the symmetrized index graph, packed B records per page; uses no logs."""
# [Paper] Section VII-A, Compared methods, "RCM and Proximity"; Table III, row RCM.
import argparse
import json
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from hyperpage.formats import read_index_meta, parse_graph


def rcm_order_from_disk_index(path, sector=4096, max_degree=64):
    """RCM order of the graph stored in the records of a DiskANN disk index.

    A record holds the int8 vector, the uint32 degree and max_degree uint32 neighbour slots; records are packed
    nnodes_per_sector per sector of the given size, after one header sector."""
    meta = read_index_meta(path)
    n, max_len, nps = meta['npts'], meta['max_node_len'], meta['nnodes_per_sector']
    nsec = -(-n // nps)
    raw = np.memmap(path, dtype=np.uint8, mode='r')
    rec = np.ascontiguousarray(raw[sector:].reshape(nsec, sector)[:, :nps * max_len]).reshape(-1, max_len)[:n]
    dim = max_len - 4 - 4 * max_degree
    deg = rec[:, dim:dim + 4].copy().view(np.uint32).ravel().astype(np.int64)
    nb = rec[:, dim + 4:dim + 4 + 4 * max_degree].copy().view(np.uint32).reshape(n, max_degree).astype(np.int64)
    used = np.arange(max_degree)[None, :] < deg[:, None]
    assert deg.max() <= max_degree and nb[used].max() < n
    rows = np.repeat(np.arange(n), deg)
    cols = nb[used]
    A = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.int8).tocsr()
    return reverse_cuthill_mckee(A, symmetric_mode=True), n, int(A.nnz)


def rcm_order_from_graph_file(path, n):
    """RCM order of the graph in a Vamana graph file (<prefix>_mem.index), with int32 column indices."""
    g_ptr, g_nbr = parse_graph(np.fromfile(path, dtype=np.uint32), n)
    A = csr_matrix((np.ones(len(g_nbr), np.int8), g_nbr.astype(np.int32),
                    g_ptr.astype(np.int32 if len(g_nbr) < 2 ** 31 else np.int64)), shape=(n, n))
    del g_ptr, g_nbr
    A = ((A + A.T.tocsr()) > 0).astype(np.int8).tocsr()
    A.sort_indices()
    return reverse_cuthill_mckee(A, symmetric_mode=True), int(A.nnz)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index-prefix', required=True,
                    help='prefix of the index files (<prefix>_disk.index, <prefix>_mem.index)')
    ap.add_argument('--graph-source', choices=('mem', 'disk'), default='mem',
                    help='read the graph from <prefix>_mem.index (mem) or from the records of <prefix>_disk.index '
                         '(disk)')
    ap.add_argument('--sector-bytes', type=int, default=4096, help='disk source: sector size of the disk index')
    ap.add_argument('--max-degree', type=int, default=64, help='disk source: neighbour slots per record, R')
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    ap.add_argument('--out', required=True, help='output layout (.npy); run details go to the same name with .json')
    a = ap.parse_args()

    t0 = time.time()
    if a.graph_source == 'disk':
        perm, n, nnz = rcm_order_from_disk_index(a.index_prefix + '_disk.index', a.sector_bytes, a.max_degree)
    else:
        n = read_index_meta(a.index_prefix + '_disk.index')['npts']
        perm, nnz = rcm_order_from_graph_file(a.index_prefix + '_mem.index', n)
    # page(v) = rank(v) // B
    rank = np.empty(n, np.int64)
    rank[perm] = np.arange(n)
    page = (rank // a.capacity).astype(np.int32)
    assert np.bincount(page).max() <= a.capacity
    np.save(a.out, page)
    info = dict(layout='RCM', graph_source=a.graph_source, capacity=a.capacity, records=int(n),
                symmetric_entries=nnz, pages=int(page.max()) + 1, seconds=time.time() - t0)
    json.dump(info, open(a.out[:-4] + '.json' if a.out.endswith('.npy') else a.out + '.json', 'w'), indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
