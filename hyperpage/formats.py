"""Readers for the files of the index and of the traversal logger, and small layout helpers."""
import numpy as np
import numba as nb


def read_index_meta(p):
    """Header of a DiskANN disk index file: number of points, dimension, entry point, record length and records per page."""
    with open(p, 'rb') as f:
        nr, nc = np.fromfile(f, dtype=np.int32, count=2)
        meta = np.fromfile(f, dtype=np.uint64, count=nr)
    keys = ['npts', 'ndims', 'medoid', 'max_node_len', 'nnodes_per_sector', 'num_frozen', 'frozen_loc', 'append_reorder']
    return {k: int(meta[i]) for i, k in enumerate(keys)} | {'raw': [int(x) for x in meta]}


@nb.njit(cache=True)
def parse_graph(raw, n):
    """Out-neighbour lists of the Vamana graph file as CSR (ptr, nbr); raw is the file read as uint32."""
    pos = 6
    deg = np.empty(n, np.int64)
    p0 = pos
    for i in range(n):
        d = raw[pos]; deg[i] = d; pos += 1 + d
    ptr = np.zeros(n + 1, np.int64)
    for i in range(n):
        ptr[i + 1] = ptr[i] + deg[i]
    nbr = np.empty(ptr[n], np.int32)
    pos = p0
    for i in range(n):
        d = raw[pos]; pos += 1
        for j in range(d):
            nbr[ptr[i] + j] = raw[pos + j]
        pos += d
    return ptr, nbr


def read_log(prefix):
    """Traversal log of the logging search application (engine/src/search_and_log.cpp).

    Per sub-query it holds the expanded nodes in expansion order (noff, nodes), the parent of each expanded node,
    the end offset of every beam round (roff, round_end), the page reads the stock engine issued (soff, sectors) and
    per-query counters."""
    with open(prefix + '.log.bin', 'rb') as f:
        nq, tn, tr, ts = np.fromfile(f, dtype=np.uint64, count=4).astype(np.int64)
        noff = np.fromfile(f, dtype=np.uint64, count=nq + 1).astype(np.int64)
        roff = np.fromfile(f, dtype=np.uint64, count=nq + 1).astype(np.int64)
        soff = np.fromfile(f, dtype=np.uint64, count=nq + 1).astype(np.int64)
        nodes = np.fromfile(f, dtype=np.uint32, count=tn).astype(np.int32)
        parents = np.fromfile(f, dtype=np.uint32, count=tn).astype(np.int64)
        round_end = np.fromfile(f, dtype=np.uint32, count=tr).astype(np.int64)
        sectors = np.fromfile(f, dtype=np.uint32, count=ts).astype(np.int64)
        n_ios = np.fromfile(f, dtype=np.uint32, count=nq).astype(np.int64)
        n_hops = np.fromfile(f, dtype=np.uint32, count=nq).astype(np.int64)
        n_cache = np.fromfile(f, dtype=np.uint32, count=nq).astype(np.int64)
        io_us = np.fromfile(f, dtype=np.float32, count=nq)
    return dict(nq=int(nq), noff=noff, roff=roff, soff=soff, nodes=nodes, parents=parents, round_end=round_end,
                sectors=sectors, n_ios=n_ios, n_hops=n_hops, n_cache=n_cache, io_us=io_us)


def read_expanded_sets(prefix):
    """Expanded sets only, as CSR (ptr, nodes) over the logged sub-queries; same file as read_log."""
    with open(prefix + '.log.bin', 'rb') as f:
        nq, tn, tr, ts = np.fromfile(f, dtype=np.uint64, count=4).astype(np.int64)
        noff = np.fromfile(f, dtype=np.uint64, count=nq + 1).astype(np.int64)
        f.seek(32 + 3 * 8 * (nq + 1))
        nodes = np.fromfile(f, dtype=np.uint32, count=tn).astype(np.int32)
    assert noff[-1] == tn
    return noff, nodes


def read_results(prefix):
    """Result identifiers of the logged sub-queries, one row of k per sub-query."""
    with open(prefix + '.res.bin', 'rb') as f:
        nq, k = np.fromfile(f, dtype=np.uint32, count=2)
        ids = np.fromfile(f, dtype=np.uint32, count=nq * k).reshape(nq, k)
    return ids


def read_ground_truth(p):
    """Exact nearest neighbours in the DiskANN ground-truth format (identifiers only)."""
    with open(p, 'rb') as f:
        n, k = np.fromfile(f, dtype=np.int32, count=2)
        ids = np.fromfile(f, dtype=np.uint32, count=n * k).reshape(n, k)
    return ids


def recall_at(res, gt, k=10):
    """Mean recall@k of result rows against ground-truth rows."""
    r = 0
    for i in range(res.shape[0]):
        r += len(set(res[i, :k].tolist()) & set(gt[i, :k].tolist()))
    return r / (res.shape[0] * k)


def layout_of_order(order, n, B):
    """Layout that packs the records in the given order, B per page: page[order[i]] = i // B."""
    page = np.empty(n, np.int32)
    page[order] = (np.arange(len(order)) // B).astype(np.int32)
    return page
