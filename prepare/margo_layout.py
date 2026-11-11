"""MARGO layout at B = page_bytes // record_bytes through MARGO's partitioner (new_mincut), converted to a layout file
(int32 page per record), and the identifier-order layout ID (page = id // B)."""
import argparse
import json
import os
import subprocess
import time

import numpy as np

from hyperpage import formats

# [Paper] Section VII-A, Implementation details and Table II caption: records are 644 bytes, so B = 12 at 8 KB


def read_partition(path, n):
    """Page of every record from MARGO's _partition.bin: header u64 {capacity, pages, points}, then per page a u32 size
    and the u32 identifiers, then u32 page per identifier (checked against the lists)."""
    with open(path, 'rb') as f:
        cap, npg, npts = np.fromfile(f, dtype=np.uint64, count=3).astype(np.int64)
        page = np.full(n, -1, np.int64)
        sizes = np.zeros(npg, np.int64)
        for p in range(npg):
            sz = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
            ids = np.fromfile(f, dtype=np.uint32, count=sz)
            page[ids] = p
            sizes[p] = sz
        id2p = np.fromfile(f, dtype=np.uint32, count=n)
    assert (page >= 0).all() and np.array_equal(page, id2p.astype(np.int64)), 'partition parse mismatch'
    return int(cap), int(npg), page, sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--partitioner', required=True, help="path of MARGO's my_gp/new_mincut")
    ap.add_argument('--index-prefix', required=True, help='prefix of the index from build_index.py')
    ap.add_argument('--data-f32', required=True,
                    help='records as float32 .fbin (base_f32.fbin, or base.fbin of DEEP1M)')
    ap.add_argument('--out', required=True, help='layout directory (LAYOUT_DIR/<workload>)')
    ap.add_argument('--work-dir', default=None, help='directory for the partitioner input and output '
                                                      '(default: <index-prefix>_margo_B<B>)')
    ap.add_argument('--page-bytes', type=int, default=8192)
    ap.add_argument('--threads', type=int, default=None, help='OMP_NUM_THREADS of the partitioner (default: inherited)')
    ap.add_argument('--expect-n', type=int, default=None, help='check: number of records')
    ap.add_argument('--expect-record-bytes', type=int, default=None, help='check: record length (644)')
    ap.add_argument('--expect-pq-bytes', type=int, default=None, help='check: PQ bytes per record (32)')
    a = ap.parse_args()
    IDX = a.index_prefix
    meta = formats.read_index_meta(IDX + '_disk.index')
    n, s = meta['npts'], meta['max_node_len']
    B = a.page_bytes // s
    if a.expect_n is not None:
        assert n == a.expect_n, (n, a.expect_n)
    if a.expect_record_bytes is not None:
        assert s == a.expect_record_bytes, (s, a.expect_record_bytes)
    if a.expect_pq_bytes is not None:
        with open(IDX + '_pq_compressed.bin', 'rb') as f:
            npq, nch = np.fromfile(f, dtype=np.int32, count=2)
        assert npq == n and nch == a.expect_pq_bytes, (int(npq), int(nch))
    os.makedirs(a.out, exist_ok=True)

    # The partitioner reads only the number of points and the records per page from the disk-index header, so a
    # header-only copy with records-per-page = B sets the page size; graph and edge weights are linked unchanged.
    wd = a.work_dir or f'{IDX}_margo_B{B}'
    os.makedirs(wd, exist_ok=True)
    pre = os.path.join(wd, 'm')
    for suf in ['_mem.index', '_mem.index.weights']:
        if not os.path.exists(pre + suf):
            os.symlink(os.path.abspath(IDX + suf), pre + suf)
    with open(IDX + '_disk.index', 'rb') as f:
        nr, nc = np.fromfile(f, dtype=np.int32, count=2)
        mt = np.fromfile(f, dtype=np.uint64, count=nr)
    mt[4] = B
    with open(pre + '_disk.index', 'wb') as f:
        np.array([nr, nc], np.int32).tofile(f)
        mt.tofile(f)
    env = dict(os.environ)
    if a.threads is not None:
        env['OMP_NUM_THREADS'] = str(a.threads)
    t = time.time()
    r = subprocess.run([a.partitioner, pre, a.data_f32], capture_output=True, text=True, env=env)
    secs = time.time() - t
    open(pre + '_partition.log', 'w').write(r.stdout + '\n' + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f'partitioner failed ({r.returncode}): ' + r.stderr[-2000:])
    cap, npg, page, sizes = read_partition(pre + '_partition.bin', n)
    assert sizes.max() <= B
    info = dict(arm='MARGO', B=int(B), cap=cap, npages=npg, secs=secs, max_page=int(sizes.max()),
                pages_not_full=int((sizes != B).sum()))
    np.save(os.path.join(a.out, f'MARGO_B{B}.npy'), page.astype(np.int32))
    json.dump(info, open(os.path.join(a.out, f'MARGO_B{B}.json'), 'w'), indent=1)
    print(json.dumps(info), flush=True)
    # identifier order: the layout of the stock engine, B consecutive identifiers per page
    np.save(os.path.join(a.out, f'ID_B{B}.npy'), (np.arange(n) // B).astype(np.int32))
    print('ID layout', n, 'records', -(-n // B), 'pages', flush=True)


if __name__ == '__main__':
    main()
