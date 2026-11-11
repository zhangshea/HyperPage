"""Relayout writer: rewrite a stock 4 KB DiskANN disk index so that page p holds the records of pi^{-1}(p) in the stock
record format, and write the node-to-(page, slot) map that the patched engine loads from <disk index>.layout."""
# [Paper] Section VI-E: a relayout writes the records in the order of pi and stores a map from node to page and slot,
# so identifiers, PQ codes and the entry point are identical across layouts.
#   - Slots inside a page follow increasing node id. Page p starts at byte UNIT * (1 + p); unused slot bytes are zero.
#   - Header (first UNIT bytes): the stock metadata with nnodes_per_sector = UNIT // max_node_len and the file-size
#     field rewritten, zero padded. For UNIT = 4096 and the layout node_id // 6 the output is byte-identical to the
#     stock file; for UNIT = 8192 and node_id // 12 it is DiskANN's stock 8 KB format.
#   - The map file holds u64 n, u32 page[n], u32 slot[n].
#   - PQ files (<prefix>_pq_*) and any <prefix>_disk.index_* side files are hard-linked (copied across devices).
#   - Every record is verified after placement, and the page capacity is checked.
# Memory: about 2.2 bytes per byte of the disk index (records and output are held in RAM).
import argparse
import glob
import json
import os
import shutil

import numpy as np

SRC_UNIT = 4096   # read unit of the stock file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stock_prefix', help='prefix of the stock index (<prefix>_disk.index, <prefix>_pq_*)')
    ap.add_argument('layout', help='layout as .npy: int32 page of every node')
    ap.add_argument('out_prefix', help='prefix of the relaid index')
    ap.add_argument('--unit', type=int, default=8192, choices=(4096, 8192), help='page size in bytes')
    a = ap.parse_args()
    src, lay, dst, UNIT = a.stock_prefix, a.layout, a.out_prefix, a.unit

    with open(src + '_disk.index', 'rb') as f:
        nr, nc = np.fromfile(f, dtype=np.int32, count=2); meta = np.fromfile(f, dtype=np.uint64, count=nr)
    npts, max_len, nps_src = int(meta[0]), int(meta[3]), int(meta[4])
    assert nr == 9 and int(meta[7]) == 0, 'only the 9-field header without reorder data is supported'
    assert nps_src >= 1 and nps_src * max_len <= SRC_UNIT, 'multi-sector nodes are not supported'
    nsec = -(-npts // nps_src)
    raw = np.memmap(src + '_disk.index', dtype=np.uint8, mode='r')
    assert len(raw) == SRC_UNIT * (1 + nsec) == int(meta[8]), 'source is not a stock 4 KB file'
    nps = UNIT // max_len                                         # page capacity B
    page = np.load(lay).astype(np.int64); assert page.shape == (npts,) and page.min() >= 0
    cnt = np.bincount(page); assert cnt.max() <= nps, f'page capacity exceeded: {cnt.max()} > {nps}'
    npages = int(page.max()) + 1
    order = np.lexsort((np.arange(npts), page))                  # by page, then node id
    start = np.zeros(npages + 1, np.int64); start[1:] = np.cumsum(cnt)
    slot = np.empty(npts, np.int64); slot[order] = np.arange(npts) - start[page[order]]
    # records in stock order: sector 1 + i // nps_src, offset (i % nps_src) * max_len
    rec = np.ascontiguousarray(raw[SRC_UNIT:].reshape(nsec, SRC_UNIT)[:, :nps_src * max_len])
    rec = rec.reshape(-1, max_len)[:npts]
    fbytes = (1 + npages) * UNIT
    out = np.zeros(fbytes, np.uint8)
    meta2 = meta.copy(); meta2[4] = nps; meta2[8] = fbytes
    out[:8] = raw[:8]; out[8:8 + 8 * nr] = meta2.view(np.uint8)
    assert not raw[8 + 8 * nr:SRC_UNIT].any(), 'unexpected bytes in the stock header sector'
    body = out[UNIT:].reshape(npages, UNIT)[:, :nps * max_len].reshape(npages, nps, max_len)
    assert np.shares_memory(body, out)
    body[page, slot] = rec
    assert np.array_equal(body[page, slot], rec), 'record verification failed'
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    out.tofile(dst + '_disk.index')
    with open(dst + '_disk.index.layout', 'wb') as f:
        np.array([npts], np.uint64).tofile(f); page.astype(np.uint32).tofile(f); slot.astype(np.uint32).tofile(f)
    for p in glob.glob(src + '_pq_*') + glob.glob(src + '_disk.index_*'):
        q = dst + p[len(src):]
        if p.endswith('.layout') or os.path.exists(q):
            continue
        try:
            os.link(p, q)
        except OSError:
            shutil.copy(p, q)
    print(json.dumps(dict(npts=npts, max_node_len=max_len, unit=UNIT, nodes_per_page=nps, npages=npages,
                          partial_pages=int((cnt < nps).sum()), empty_pages=int((cnt == 0).sum()),
                          file_bytes=int(fbytes), layout=os.path.basename(lay))))


if __name__ == '__main__':
    main()
