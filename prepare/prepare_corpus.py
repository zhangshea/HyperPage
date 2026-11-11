"""Corpus texts as texts.bin (UTF-8, concatenated) and offsets.npy (int64, n + 1): the OpenResearcher slice of W1, and
the DPR passages (a sorted uniform sample, or all passages in file order)."""
import argparse
import csv
import glob
import gzip
import os

import numpy as np

DPR_PASSAGES = 21015324


def write_texts(out, texts_iter, n_expected):
    offs = np.zeros(n_expected + 1, np.int64)
    i = 0
    with open(os.path.join(out, 'texts.bin'), 'wb', buffering=1 << 24) as f:
        for t in texts_iter:
            b = t.encode('utf-8', errors='replace')
            f.write(b)
            offs[i + 1] = offs[i] + len(b)
            i += 1
    assert i == n_expected, (i, n_expected)
    np.save(os.path.join(out, 'offsets.npy'), offs)
    print('texts', n_expected, 'bytes', int(offs[-1]))


def openresearcher(src, n, max_chars):
    """Rows 0..n-1 of OpenResearcher-Corpus in docid order, first max_chars characters of `text`."""
    import pyarrow.parquet as pq
    k = 0
    for f in sorted(glob.glob(os.path.join(src, 'train-*.parquet'))):
        t = pq.read_table(f, columns=['docid', 'text'])
        for d, x in zip(t.column('docid').to_pylist(), t.column('text').to_pylist()):
            if k >= n:
                return
            assert int(d) == k, (d, k)
            yield (x or '')[:max_chars]
            k += 1


def dpr(src, mark):
    """DPR passages as title + newline + text, in file order; with a mask, only the marked rows."""
    csv.field_size_limit(1 << 30)
    i = -1
    with gzip.open(src, 'rt', encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter='\t')
        hdr = next(r)
        assert hdr == ['id', 'text', 'title'], hdr
        for i, row in enumerate(r):
            if mark is None or mark[i]:
                yield row[2] + '\n' + row[1]
    assert i == DPR_PASSAGES - 1, i


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', choices=['openresearcher', 'dpr'])
    ap.add_argument('--src', required=True, help='OpenResearcher-Corpus parquet directory, or psgs_w100.tsv.gz')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--n', type=int, default=None,
                    help='openresearcher: rows to keep (default 1,000,000); dpr: sample size (default: all passages)')
    ap.add_argument('--max-chars', type=int, default=4000, help='openresearcher: characters kept per document')
    ap.add_argument('--seed', type=int, default=0x13528a6, help='dpr: seed of the uniform sample')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.source == 'openresearcher':
        n = a.n or 1_000_000
        write_texts(a.out, openresearcher(a.src, n, a.max_chars), n)
    elif a.n is None:
        write_texts(a.out, dpr(a.src, None), DPR_PASSAGES)
    else:
        # sorted uniform sample of passage rows; sel_rows.npy maps text row i to the passage row of the file
        sel = np.sort(np.random.default_rng(a.seed).choice(DPR_PASSAGES, a.n, replace=False))
        np.save(os.path.join(a.out, 'sel_rows.npy'), sel)
        mark = np.zeros(DPR_PASSAGES, bool)
        mark[sel] = True
        write_texts(a.out, dpr(a.src, mark), a.n)


if __name__ == '__main__':
    main()
