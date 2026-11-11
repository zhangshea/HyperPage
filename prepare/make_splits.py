"""Training draws, validation and test splits of every workload as scopes (groups, sessions, proxy groups or batches)
of sub-queries, plus the step windows and the shuffled control of W3; writes splits.npz and the needed strings."""
import argparse
import gzip
import json
import os

import numpy as np

# [Paper] Section VII-A, Datasets and Table II: splits of W3 (training draws by step, validation, test), W1, W2, W4 and
# DEEP1M


def read_calls(D):
    qs = json.load(open(os.path.join(D, 'qstr.json')))
    qid_of = {q: i for i, q in enumerate(qs)}
    rows = [json.loads(line) for line in gzip.open(os.path.join(D, 'calls.jsonl.gz'), 'rt', encoding='utf-8')]
    return qs, qid_of, rows


def split_w1(D, a):
    """W1: question ids permuted with the seed; the first n_train form the training pool, the next n_val validation,
    the rest test. A scope is one trajectory (its calls in order). Draw r takes, for each training question, the
    trajectory at position r of default_rng([seed, qid]).permutation(sorted seeds of that question)."""
    qs, qid_of, rows = read_calls(D)
    traj = {}
    for s, q, st, ci, text in rows:
        traj.setdefault((s, q), []).append(qid_of[text])
    qids = np.array(sorted({q for (_, q) in traj}))
    perm = np.random.default_rng(a.seed).permutation(len(qids))
    qp = qids[perm]
    nt, nv = a.w1_train, a.w1_train + a.w1_val
    split_q = {'train': qp[:nt], 'val': qp[nt:nv], 'test': qp[nv:]}
    seeds_of = {}
    for (s, q) in traj:
        seeds_of.setdefault(q, []).append(s)
    out = {}
    for k in ('val', 'test'):
        sq = set(split_q[k].tolist())
        out[k] = sorted([key for key in traj if key[1] in sq], key=lambda t: (t[1], t[0]))
    for r in range(3):
        keys = []
        for q in sorted(split_q['train'].tolist()):
            ss = sorted(seeds_of[q])
            pr = np.random.default_rng([a.seed, int(q)]).permutation(len(ss))
            keys.append((ss[pr[r]], q))
        out[f'train{r}'] = keys
    res, meta = {}, {}
    for k, keys in out.items():
        bl = [traj[key] for key in keys]
        res[k] = bl
        meta[k] = dict(n_batches=len(bl), n_calls=int(sum(len(b) for b in bl)), n_units=len({key[1] for key in keys}))
        np.save(os.path.join(D, f'meta_{k}.npy'), np.array(keys, np.int64))
    unit = {k: np.array([key[1] for key in out[k]], np.int64) for k in out}
    return qs, res, unit, meta


def split_w2(D, a):
    """W2: a scope is a (step, prompt) group, sub-queries ordered by (rollout line, tool call, list position). Steps
    1..train_hi form the draws ((s - 1) mod 3 = r), then validation and test; held-out groups whose prompt occurs in a
    training step are removed."""
    qs, qid_of, rows = read_calls(D)
    grp = {}
    for s, ph, li, ti, qi, text in rows:
        grp.setdefault((s, ph), []).append((li, ti, qi, qid_of[text]))
    keys_all = sorted(grp, key=lambda t: (t[0], t[1]))

    def sel(f):
        return [k for k in keys_all if f(k[0])]
    th, vh = a.w2_train_hi, a.w2_val_hi
    out = {f'train{r}': sel(lambda s, r=r: s <= th and (s - 1) % 3 == r) for r in range(3)}
    out.update(val=sel(lambda s: th < s <= vh), test=sel(lambda s: s > vh))
    train_prompts = {k[1] for kk in ('train0', 'train1', 'train2') for k in out[kk]}
    excl = {}
    for k in ('val', 'test'):
        before = len(out[k])
        out[k] = [key for key in out[k] if key[1] not in train_prompts]
        excl[k] = before - len(out[k])
    res, meta = {}, {}
    for k, keys in out.items():
        res[k] = [[x[3] for x in sorted(grp[key])] for key in keys]
        meta[k] = dict(n_batches=len(keys), n_calls=int(sum(len(grp[key]) for key in keys)), n_units=len(keys),
                       excluded_prompt_overlap=excl.get(k, 0))
        np.save(os.path.join(D, f'meta_{k}.npy'),
                np.array([[s, int(ph, 16) & ((1 << 62) - 1)] for s, ph in keys], np.int64))
    unit = {k: np.arange(len(out[k]), dtype=np.int64) for k in out}
    return qs, res, unit, meta


def groups_of(rows, qid_of):
    grp = {}
    for src, s, k, li, ti, qi, text in rows:
        grp.setdefault((src, s, k), []).append((li, ti, qi, qid_of[text]))
    return grp


def split_w3(D, a):
    """W3: a scope is a (step, prompt) group. train<r>: outcome-run steps train_lo..400 with (s - 1) mod 3 = r;
    val and test: process-run step ranges; s1_40 and s201_240: the process-run step windows before validation and
    between test and training. Groups listed in drop.json are removed. The shuffled control (<key>_shuf) permutes,
    per step, all sub-queries across that step's groups with default_rng(shuffle_seed + j) and cuts them into blocks
    of the true group sizes in group order."""
    qs, qid_of, rows = read_calls(D)
    drop = json.load(open(os.path.join(D, 'drop.json')))
    grp = groups_of(rows, qid_of)
    dset = {(int(s), k) for s, k in drop['drop_groups']}

    def sel(src, lo, hi, mod=None):
        ks = sorted(k for k in grp if k[0] == src and lo <= k[1] <= hi and (mod is None or (k[1] - 1) % 3 == mod))
        return [k for k in ks if not (src == 'perturnnorm' and (k[1], k[2]) in dset)]
    keys = {f'train{r}': sel('4gpu', a.w3_train[0], a.w3_train[1], r) for r in range(3)}
    keys.update(val=sel('perturnnorm', *a.w3_val), test=sel('perturnnorm', *a.w3_test),
                s1_40=sel('perturnnorm', 1, a.w3_val[0] - 1), s201_240=sel('perturnnorm', a.w3_test[1] + 1,
                                                                          a.w3_train[0] - 1))
    res = {k: [[x[3] for x in sorted(grp[key])] for key in v] for k, v in keys.items()}
    # [Paper] Section VII-C: refining on sub-queries shuffled across the groups of a step
    for j, k in enumerate(['train0', 'train1', 'train2', 'val', 'test']):
        rng = np.random.default_rng(a.shuffle_seed + j)
        steps = sorted({key[1] for key in keys[k]})
        by_step = {s: [i for i, key in enumerate(keys[k]) if key[1] == s] for s in steps}
        new = [None] * len(keys[k])
        for s in steps:
            idx = by_step[s]
            allq = np.concatenate([np.asarray(res[k][i], np.int64) for i in idx]) if idx else np.zeros(0, np.int64)
            allq = allq[rng.permutation(len(allq))]
            c = 0
            for i in idx:
                n_ = len(res[k][i])
                new[i] = allq[c:c + n_]
                c += n_
        res[k + '_shuf'] = new
    for k, v in keys.items():
        np.save(os.path.join(D, f'meta_{k}.npy'),
                np.array([[0 if src == '4gpu' else 1, s, int(kk, 16) & ((1 << 62) - 1)] for src, s, kk in v], np.int64))
    return qs, res


def split_w4(D, a):
    """W4: train<r>: all outcome-run steps with (s - 1) mod 3 = r, scope = (step, prompt) group; held-out scope = one
    OLMo-3 question over all checkpoints (proxy group), sub-queries ordered by (checkpoint, tool call, list position).
    Questions not in drop.json are permuted with the seed: the first 20% validation, the rest test."""
    qs, qid_of, rows = read_calls(D)
    drop = json.load(open(os.path.join(D, 'drop.json')))
    grp = groups_of(rows, qid_of)
    dset = set(drop['drop_index'])
    keys = {f'train{r}': sorted(k for k in grp if k[0] == '4gpu' and (k[1] - 1) % 3 == r) for r in range(3)}
    eval_g = {}
    for (src, ck, k), v in grp.items():
        if src == 'olmo3':
            for li, ti, qi, sid in v:
                eval_g.setdefault(li, []).append((ck, ti, qi, sid))   # li is the question index
    qidx = np.array(sorted(i for i in eval_g if i not in dset))
    perm = qidx[np.random.default_rng(a.w4_seed).permutation(len(qidx))]
    nv = int(round(a.w4_val_share * len(perm)))
    res = {k: [[x[3] for x in sorted(grp[key])] for key in v] for k, v in keys.items()}
    res['val'] = [[x[3] for x in sorted(eval_g[i])] for i in sorted(perm[:nv])]
    res['test'] = [[x[3] for x in sorted(eval_g[i])] for i in sorted(perm[nv:])]
    np.save(os.path.join(D, 'meta_val.npy'), np.array(sorted(perm[:nv])))
    np.save(os.path.join(D, 'meta_test.npy'), np.array(sorted(perm[nv:])))
    return qs, res


def pack(bl):
    bl = [np.asarray(b, np.int64) for b in bl]
    ptr = np.zeros(len(bl) + 1, np.int64)
    ptr[1:] = np.cumsum([len(b) for b in bl])
    return ptr, (np.concatenate(bl) if bl else np.zeros(0, np.int64))


def write_text_splits(D, w, a):
    if w in ('w1', 'w2'):
        qs, res, unit, meta = (split_w1 if w == 'w1' else split_w2)(D, a)
        arrs = {}
        for k, bl in res.items():
            arrs[k + '_ptr'], arrs[k + '_q'] = pack(bl)
        for k, u in unit.items():
            arrs[k + '_unit'] = u
    else:
        qs, res = (split_w3 if w == 'w3' else split_w4)(D, a)
        arrs, meta = {}, {}
        for k, bl in res.items():
            arrs[k + '_ptr'], arrs[k + '_q'] = pack(bl)
            arrs[k + '_unit'] = np.arange(len(bl))
            meta[k] = dict(n_batches=len(bl), n_calls=int(arrs[k + '_ptr'][-1]),
                           empty_batches=int(sum(len(b) == 0 for b in bl)))
    np.savez(os.path.join(D, 'splits.npz'), **arrs)
    # every distinct string that some split uses; the logging run traverses exactly these
    need = np.unique(np.concatenate([arrs[k + '_q'] for k in res]))
    np.save(os.path.join(D, 'queries_needed.npy'), need)
    meta['n_distinct_needed'] = int(len(need))
    meta['n_distinct_all'] = len(qs)
    json.dump(meta, open(os.path.join(D, 'splits_meta.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


def write_deep_splits(D, a):
    """DEEP1M random batches with the sizes of a text workload (W3). Query k is held-out DEEP row k of the logged
    pool (see deep_prepare.py). Training draw r: the first N_r rows of pool block r, cut in order into batches whose
    sizes are draw r's batch sizes permuted with default_rng(seed + r). Validation / test: sizes drawn i.i.d. from
    the matched validation / test batch sizes until the rows run out; the last partial batch is dropped."""
    A = np.load(a.match)
    arrs, m = {}, {}
    for r in range(3):
        sizes = np.diff(A[f'train{r}_ptr'])
        sizes = sizes[np.random.default_rng(a.seed + r).permutation(len(sizes))]
        N = int(sizes.sum())
        assert N <= a.deep_block, (N, a.deep_block)
        ids = r * a.deep_block + np.arange(N, dtype=np.int64)
        ptr = np.zeros(len(sizes) + 1, np.int64)
        ptr[1:] = np.cumsum(sizes)
        arrs[f'train{r}_ptr'] = ptr
        arrs[f'train{r}_q'] = ids
        arrs[f'train{r}_unit'] = np.arange(len(sizes))
        m[f'train{r}'] = dict(n_batches=len(sizes), n_calls=N)
    for split, k0, nr, seed in (('val', a.deep_val[0], a.deep_val[1], a.deep_val_seed),
                                ('test', a.deep_test[0], a.deep_test[1], a.deep_test_seed)):
        pool = np.diff(A[f'{split}_ptr'])
        rng = np.random.default_rng(seed)
        sizes = []
        tot = 0
        while True:
            s = int(pool[rng.integers(len(pool))])
            if tot + s > nr:
                break
            sizes.append(s)
            tot += s
        sizes = np.array(sizes, np.int64)
        ptr = np.zeros(len(sizes) + 1, np.int64)
        ptr[1:] = np.cumsum(sizes)
        arrs[f'{split}_ptr'] = ptr
        arrs[f'{split}_q'] = np.arange(k0, k0 + tot, dtype=np.int64)
        arrs[f'{split}_unit'] = np.arange(len(sizes))
        m[split] = dict(n_batches=len(sizes), n_calls=int(tot))
    np.savez(os.path.join(D, 'splits.npz'), **arrs)
    used = np.unique(np.concatenate([arrs[k] for k in arrs if k.endswith('_q')]))
    logged = np.load(os.path.join(D, 'queries_needed.npy'))
    assert np.isin(used, logged).all(), 'a split uses a query outside the logged pool'
    m['n_used'] = int(len(used))
    json.dump(m, open(os.path.join(D, 'splits_meta.json'), 'w'), indent=1)
    print(json.dumps(m, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('workload', choices=['w1', 'w2', 'w3', 'w4', 'deep'])
    ap.add_argument('--data-dir', required=True,
                    help='DATA_DIR/<workload>: qstr.json, calls.jsonl.gz (and drop.json for W3, W4); '
                         'for deep: queries_needed.npy from deep_prepare.py')
    ap.add_argument('--seed', type=int, default=0x13528a6,
                    help='W1: question permutation and draw seed; deep: base seed of the training-size permutations')
    ap.add_argument('--w1-train', type=int, default=3661, help='W1: questions in the training pool')
    ap.add_argument('--w1-val', type=int, default=610, help='W1: validation questions (the rest are test)')
    ap.add_argument('--w2-train-hi', type=int, default=240, help='W2: last training step')
    ap.add_argument('--w2-val-hi', type=int, default=280, help='W2: last validation step')
    ap.add_argument('--w3-train', type=int, nargs=2, default=[241, 400], help='W3: outcome-run training steps')
    ap.add_argument('--w3-val', type=int, nargs=2, default=[41, 70], help='W3: process-run validation steps')
    ap.add_argument('--w3-test', type=int, nargs=2, default=[71, 200], help='W3: process-run test steps')
    ap.add_argument('--shuffle-seed', type=int, default=0x13528ed, help='W3: base seed of the shuffled control')
    ap.add_argument('--w4-seed', type=int, default=0x13528a7, help='W4: seed of the question permutation')
    ap.add_argument('--w4-val-share', type=float, default=0.2, help='W4: share of questions in validation')
    ap.add_argument('--match', default=None, help='deep: splits.npz of the workload whose sizes are matched (W3)')
    ap.add_argument('--deep-block', type=int, default=400000, help='deep: pool rows per training draw')
    ap.add_argument('--deep-val', type=int, nargs=2, default=[1268000, 20000], help='deep: first query and rows')
    ap.add_argument('--deep-test', type=int, nargs=2, default=[1368000, 200000], help='deep: first query and rows')
    ap.add_argument('--deep-val-seed', type=int, default=0x13528ba)
    ap.add_argument('--deep-test-seed', type=int, default=0x13528bb)
    a = ap.parse_args()
    if a.workload == 'deep':
        write_deep_splits(a.data_dir, a)
    else:
        write_text_splits(a.data_dir, a.workload, a)


if __name__ == '__main__':
    main()
