"""Write a layout (page of every record) in the partition format of MARGO's index_relayout tool, which then writes the
8 KB page-search index file in that layout: u64 C, u64 pages, u64 n; per page u32 size, u32 ids; u32 id2page[n]."""
# [Paper] Section VII-B, MARGO's page-search engine: the page-search layouts of MARGO and HyperPage share one file
# format. Records inside a page follow increasing id; the written file is read back and checked.
import argparse

import numpy as np


def read_partition(p):
    h = np.fromfile(p, np.uint64, count=3); C, npg, n = (int(x) for x in h)
    raw = np.fromfile(p, np.uint32, offset=24)
    pages, i = [], 0
    for _ in range(npg):
        s = int(raw[i]); pages.append(raw[i + 1:i + 1 + s].copy()); i += 1 + s
    id2page = raw[i:i + n]
    assert len(id2page) == n, (len(id2page), n)
    return C, pages, id2page


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('layout', help='layout (.npy, page of every record)')
    ap.add_argument('out', help='partition file to write')
    ap.add_argument('--capacity', type=int, default=12, help='records per page, B')
    a = ap.parse_args()
    page = np.asarray(np.load(a.layout), np.int64); n = len(page); npg = int(page.max()) + 1; C = a.capacity
    order = np.lexsort((np.arange(n), page)); cnt = np.bincount(page, minlength=npg)
    assert cnt.max() <= C
    with open(a.out, 'wb') as f:
        np.array([C, npg, n], np.uint64).tofile(f)
        pos = 0
        for p in range(npg):
            ids = order[pos:pos + cnt[p]]; pos += cnt[p]
            np.array([len(ids)], np.uint32).tofile(f); ids.astype(np.uint32).tofile(f)
        page.astype(np.uint32).tofile(f)
    C2, pages, id2 = read_partition(a.out)
    assert C2 == C and len(pages) == npg and np.array_equal(id2, page.astype(np.uint32))
    for p in range(0, npg, max(1, npg // 1000)):
        assert np.all(page[pages[p]] == p)
    print(f'partition {a.out}: {npg} pages, n={n}, max fill {cnt.max()}, empty pages {(cnt == 0).sum()}')


if __name__ == '__main__':
    main()
