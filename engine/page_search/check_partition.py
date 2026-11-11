"""Checks of the page-search inputs: a partition file holds the pages of a layout up to page renumbering (same-pages),
and the index file written by MARGO's index_relayout holds, page by page, the stock records it lists (file)."""
# [Paper] Section VII-B, MARGO's page-search engine: the 8 KB page-search files of MARGO's and HyperPage's layouts
# (Table V). Exit status 0 when the check passes, 1 otherwise.
import argparse
import os
import sys

import numpy as np

from write_partition import read_partition


def page_sets(page):
    """The set of record sets of the non-empty pages, each in increasing id."""
    page = np.asarray(page, np.int64); order = np.lexsort((np.arange(len(page)), page))
    cnt = np.bincount(page); starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return {tuple(order[s:s + c]) for s, c in zip(starts, cnt) if c > 0}


def same_pages(a):
    lay = np.load(a.layout); C, pages, id2 = read_partition(a.partition)
    sa, sb = page_sets(lay), page_sets(id2)
    ok = sa == sb
    print(f"same pages {'PASS' if ok else 'FAIL'}: layout pages {len(sa)}, partition pages {len(sb)}, "
          f"common {len(sa & sb)}")
    return ok


def index_file(a):
    C, pages, id2 = read_partition(a.partition)
    meta = np.fromfile(a.stock, np.uint32, count=2); nr = int(meta[0])
    m = np.fromfile(a.stock, np.uint64, offset=8, count=nr)   # npts, dim, medoid, max_node_len, nnodes_per_sector, ...
    node_len, nps = int(m[3]), int(m[4])
    assert nps == a.stock_page // node_len, (nps, node_len)
    # one header page, then one page per layout page
    sz = os.path.getsize(a.index); expected = len(pages) * a.page + a.page
    mn = np.fromfile(a.index, np.uint64, offset=8, count=nr)
    print(f'index file size {sz} (expected {expected}), records per page {int(mn[4])}, record length {int(mn[3])}')
    ok = sz == expected and int(mn[4]) == C
    # sampled pages: every slot must hold, byte for byte, the stock record of the id the partition lists there
    rng = np.random.default_rng(a.seed); bad = 0
    with open(a.stock, 'rb') as fs, open(a.index, 'rb') as fn:
        for p in rng.choice(len(pages), a.samples, replace=False):
            fn.seek(a.page + int(p) * a.page); buf = fn.read(a.page)
            for j, v in enumerate(pages[p]):
                fs.seek((int(v) // nps + 1) * a.stock_page + (int(v) % nps) * node_len); rec = fs.read(node_len)
                if buf[j * node_len:(j + 1) * node_len] != rec:
                    bad += 1
    ok = ok and bad == 0
    print(f"index file {'PASS' if ok else 'FAIL'}: {a.samples} pages sampled, mismatched records {bad}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('same-pages', help='partition file and layout hold the same pages')
    s.add_argument('layout', help='layout (.npy, page of every record)')
    s.add_argument('partition', help="partition file (MARGO's own or write_partition.py's)")
    f = sub.add_parser('file', help="page-search index file against the stock index and the partition")
    f.add_argument('stock', help='stock index file (<stock prefix>_disk.index)')
    f.add_argument('partition', help='partition file given to index_relayout')
    f.add_argument('index', help='page-search index file written by index_relayout (m_disk.index)')
    f.add_argument('--samples', type=int, default=100, help='pages compared')
    f.add_argument('--seed', type=int, default=23, help='seed of the sampled pages')
    f.add_argument('--stock-page', type=int, default=4096, help='page size of the stock index')
    f.add_argument('--page', type=int, default=8192, help='page size of the page-search index')
    a = ap.parse_args()
    sys.exit(0 if (same_pages(a) if a.cmd == 'same-pages' else index_file(a)) else 1)


if __name__ == '__main__':
    main()
