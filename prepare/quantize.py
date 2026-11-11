"""int8 records and query vectors, x8 = clip(rint(gamma x), -127, 127) with one corpus scale gamma; records are stored
in a seeded random node order (node j = text row perm[j]). Also the n-record samples of the full-corpus records."""
import argparse
import json
import os

import numpy as np

DPR_PASSAGES = 21015324
CHUNK = 1 << 20


def to_int8(v):
    """Scaled values rounded to the nearest integer and clipped to [-127, 127]."""
    return np.clip(np.rint(v), -127, 127).astype(np.int8)


def write_header(f, n, d):
    np.array([n, d], np.int32).tofile(f)


def load_doc_embeddings(d, nshards):
    """Corpus embeddings as float16 (n, 384): doc_emb.f16.npy, or the raw shards doc_s<r>.f16 written by embed.py."""
    if nshards is None:
        return np.load(os.path.join(d, 'doc_emb.f16.npy'))
    off = np.load(os.path.join(d, 'offsets.npy'), mmap_mode='r')
    n = len(off) - 1
    x = np.empty((n, 384), np.float16)
    for r in range(nshards):
        lo, hi = n * r // nshards, n * (r + 1) // nshards
        assert int(open(os.path.join(d, f'doc_s{r}.done')).read()) == hi, f'shard {r} incomplete'
        x[lo:hi] = np.fromfile(os.path.join(d, f'doc_s{r}.f16'), dtype=np.float16).reshape(hi - lo, 384)
    return x


def corpus(a):
    D = a.data_dir
    x = load_doc_embeddings(D, a.nshards)
    n, d = x.shape
    if a.nshards is not None:
        # shards are written independently: check that every one holds normalised embeddings
        nrm = np.sqrt((x[::1000].astype(np.float32) ** 2).sum(1))
        assert abs(nrm.mean() - 1) < 1e-2 and nrm.min() > 0.9, 'embeddings are not L2-normalised'
    info = dict(n=n, d=d)
    if a.gamma_from:
        g = json.load(open(a.gamma_from))['gamma']
        info.update(gamma_from=a.gamma_from, gamma=g)
    else:
        # gamma = 127 / Q_0.9999(|x_ij|) over all corpus entries
        qv = float(np.quantile(np.abs(x.astype(np.float32)).ravel(), 0.9999))
        g = 127.0 / qv
        info.update(q_abs_9999=qv, gamma=g)
    perm = np.random.default_rng(a.seed).permutation(n)
    np.save(os.path.join(D, 'perm.npy'), perm.astype(np.int64))
    clip = 0
    # base.bin is the int8 data file of the index builder; base_f32.fbin holds the same values as float32 for MARGO
    with open(os.path.join(D, 'base.bin'), 'wb') as f8, open(os.path.join(D, 'base_f32.fbin'), 'wb') as f32:
        write_header(f8, n, d)
        write_header(f32, n, d)
        for c in range(0, n, CHUNK):
            v = x[perm[c:c + CHUNK]].astype(np.float32) * g
            clip += int((np.abs(v) > 127).sum())
            v8 = to_int8(v)
            v8.tofile(f8)
            v8.astype(np.float32).tofile(f32)
    info['clip_frac_docs'] = clip / (n * d)
    if os.path.exists(os.path.join(D, 'q_emb.f16.npy')):
        quantize_queries(D, g, info)
    json.dump(info, open(os.path.join(D, 'quant.json'), 'w'), indent=1)
    print(info)


def quantize_queries(D, g, info):
    q = np.load(os.path.join(D, 'q_emb.f16.npy')).astype(np.float32)
    np.save(os.path.join(D, 'qx8.npy'), to_int8(g * q))
    info.update(n_queries=int(q.shape[0]), clip_frac_queries=float((np.abs(g * q) > 127).mean()))


def queries(a):
    """Query vectors of a workload that shares the corpus of another: gamma is read from that corpus's quant.json."""
    g = json.load(open(a.gamma_from))['gamma']
    info = dict(gamma_from=a.gamma_from, gamma=g)
    quantize_queries(a.data_dir, g, info)
    json.dump(info, open(os.path.join(a.data_dir, 'quant.json'), 'w'), indent=1)
    print(info)


def sample(a):
    """An n-record uniform sample of the full corpus with the convention of the 1M sample: rows
    sel = sorted(choice(N, n)), node j = sel[perm[j]]. The int8 records are gathered from the full-corpus base.bin
    through the inverse of its node permutation, so nothing is re-embedded."""
    n, d = a.n, 384
    full = a.full_dir
    g = json.load(open(os.path.join(full, 'quant.json')))['gamma']
    sel = np.sort(np.random.default_rng(a.seed).choice(DPR_PASSAGES, n, replace=False))
    perm_full = np.load(os.path.join(full, 'perm.npy'))
    inv = np.empty(DPR_PASSAGES, np.int64)
    inv[perm_full] = np.arange(DPR_PASSAGES)
    with open(os.path.join(full, 'base.bin'), 'rb') as f:
        hn, hd = np.fromfile(f, np.int32, 2)
        assert (hn, hd) == (DPR_PASSAGES, d)
        b = np.fromfile(f, np.int8, DPR_PASSAGES * d).reshape(DPR_PASSAGES, d)
    perm = np.random.default_rng(a.seed).permutation(n)
    rows = sel[perm]
    x8 = b[inv[rows]]
    D = a.data_dir
    os.makedirs(D, exist_ok=True)
    with open(os.path.join(D, 'base.bin'), 'wb') as f:
        write_header(f, n, d)
        x8.tofile(f)
    with open(os.path.join(D, 'base_f32.fbin'), 'wb') as f:
        write_header(f, n, d)
        for c in range(0, n, CHUNK):
            x8[c:c + CHUNK].astype(np.float32).tofile(f)
    np.save(os.path.join(D, 'perm.npy'), perm.astype(np.int64))
    np.save(os.path.join(D, 'sel.npy'), sel.astype(np.int64))
    info = dict(n=n, d=d, gamma=g, source='full-corpus int8 records gathered by row')
    json.dump(info, open(os.path.join(D, 'quant.json'), 'w'), indent=1)
    print(json.dumps(info), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['corpus', 'queries', 'sample'])
    ap.add_argument('--data-dir', required=True, help='workload directory (reads embeddings, writes int8 files)')
    ap.add_argument('--gamma-from', default=None,
                    help='quant.json whose gamma is reused (corpus: optional; queries: required)')
    ap.add_argument('--nshards', type=int, default=None, help='corpus: read embeddings from this many shards')
    ap.add_argument('--seed', type=int, default=0x13528a6, help='seed of the node permutation and of the sample')
    ap.add_argument('--full-dir', help='sample: directory of the full-corpus base.bin, perm.npy, quant.json')
    ap.add_argument('--n', type=int, help='sample: number of records')
    a = ap.parse_args()
    {'corpus': corpus, 'queries': queries, 'sample': sample}[a.mode](a)


if __name__ == '__main__':
    main()
