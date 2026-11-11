"""Arrival metadata of one split for the replay: step, turn, rollout, position, group and string of every sub-query."""
# [Paper] Section VII-A, Datasets and Metrics: W3 groups are the (step, prompt) groups of the process-reward run; W4
# proxy steps are (checkpoint, block of questions); W2 uses the outcome-reward run alone; W1 cuts trajectories into
# steps; DEEP1M cuts its random batches, in split order, into equal steps.
# The output is aligned with the per-query sets file of the split and is read by replay_hybrid.py.
import argparse
import gzip
import json

import numpy as np

# step windows (first, last) of the named splits
W3_WINDOWS = {'val': (41, 70), 'test': (71, 200), 's1_40': (1, 40), 's201_240': (201, 240)}
W2_WINDOWS = {'val': (241, 280), 'test': (281, 400)}
DEEP_STEPS = {'val': 30, 'test': 130}   # DEEP1M: steps of the validation and test splits, as W3


def read_calls(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def w3(a, n_calls, lo, hi):
    """Groups (step, prompt) of the process-reward run inside [lo, hi]; inside a group by (rollout, turn, position).
    Process-run groups listed in drop.json are skipped, as make_splits.py removes them from the splits."""
    dropped = set()
    if a.drop:
        dropped = {(int(s), k) for s, k in json.load(open(a.drop))['drop_groups']}
    groups = {}
    for src, s, k, li, ti, qi, q in read_calls(a.calls):
        if src == a.source and lo <= s <= hi and not (src == 'perturnnorm' and (s, k) in dropped):
            groups.setdefault((s, k), []).append((li, ti, qi))
    cols = {x: np.zeros(n_calls, np.int64) for x in ('step', 'turn', 'rollout', 'position', 'group')}
    c = 0
    for g, key in enumerate(sorted(groups)):
        for li, ti, qi in sorted(groups[key]):
            assert c < n_calls, 'more sub-queries in the calls file than in the split'
            cols['step'][c] = key[0] - lo; cols['rollout'][c] = li; cols['turn'][c] = ti; cols['position'][c] = qi
            cols['group'][c] = g
            c += 1
    sizes = [len(groups[key]) for key in sorted(groups)]
    return cols, c, sizes, hi - lo + 1


def w2(a, n_calls, lo, hi):
    """Groups (step, prompt) of the outcome-reward run inside [lo, hi], without prompts seen in a training step."""
    groups = {}
    for s, ph, li, ti, qi, q in read_calls(a.calls):
        groups.setdefault((s, ph), []).append((li, ti, qi))
    seen = {key[1] for key in groups if key[0] <= a.train_last_step}
    keys = [key for key in sorted(groups) if lo <= key[0] <= hi and key[1] not in seen]
    cols = {x: np.zeros(n_calls, np.int64) for x in ('step', 'turn', 'rollout', 'position', 'group')}
    c = 0
    for g, key in enumerate(keys):
        for li, ti, qi in sorted(groups[key]):
            assert c < n_calls, 'more sub-queries in the calls file than in the split'
            cols['step'][c] = key[0] - lo; cols['rollout'][c] = li; cols['turn'][c] = ti; cols['position'][c] = qi
            cols['group'][c] = g
            c += 1
    return cols, c, [len(groups[key]) for key in keys], hi - lo + 1


def w4(a, n_calls):
    """Proxy groups: one question over all checkpoints. A step is one checkpoint and one block of questions."""
    strings = json.load(open(a.strings))
    string_id = {q: i for i, q in enumerate(strings)}
    keep = set(np.load(a.questions).tolist())
    groups = {}
    for src, ck, k, li, ti, qi, q in read_calls(a.calls):
        if src == a.source and li in keep:
            groups.setdefault(li, []).append((ck, ti, qi, string_id[q]))
    checkpoints = sorted({x[0] for v in groups.values() for x in v})
    ck_index = {ck: i for i, ck in enumerate(checkpoints)}
    q_rank = {li: i for i, li in enumerate(sorted(groups))}
    n_blocks = -(-len(groups) // a.block)
    cols = {x: np.zeros(n_calls, np.int64) for x in ('step', 'turn', 'rollout', 'position', 'group')}
    c = 0
    for li in sorted(groups):
        # calls of a question in (checkpoint, turn, position, string) order; the step is (checkpoint, question block)
        for ck, ti, qi, sid in sorted(groups[li]):
            assert c < n_calls, 'more sub-queries in the calls file than in the split'
            cols['step'][c] = ck_index[ck] * n_blocks + q_rank[li] // a.block
            cols['rollout'][c] = li; cols['turn'][c] = ti; cols['position'][c] = qi; cols['group'][c] = q_rank[li]
            c += 1
    n_steps = len(checkpoints) * n_blocks
    per_step = np.bincount(cols['step'][:c], minlength=n_steps)
    print(f'{len(groups)} questions, {len(checkpoints)} checkpoints, {n_blocks} blocks, {n_steps} steps, '
          f'sub-queries per step min {per_step.min()} median {int(np.median(per_step))} max {per_step.max()}')
    return cols, c, [len(groups[li]) for li in sorted(groups)], n_steps


def w1(a, n_calls, group_ptr):
    """Trajectories in split order; every per_step trajectories form a step, and a call's turn is its position."""
    cols = {x: np.zeros(n_calls, np.int64) for x in ('step', 'turn', 'rollout', 'position', 'group')}
    c = 0
    for b in range(len(group_ptr) - 1):
        for j in range(group_ptr[b], group_ptr[b + 1]):
            cols['step'][c] = b // a.per_step; cols['rollout'][c] = b; cols['turn'][c] = j - group_ptr[b]
            cols['group'][c] = b
            c += 1
    return cols, c, np.diff(group_ptr).tolist(), -(-(len(group_ptr) - 1) // a.per_step)


def deep(a, n_calls, group_ptr):
    """Sub-queries in split order, cut into n_steps steps of equal size; the replay keeps the split order."""
    n_steps = a.n_steps or DEEP_STEPS[a.split]
    c = np.arange(n_calls, dtype=np.int64)
    cols = {x: np.zeros(n_calls, np.int64) for x in ('step', 'turn', 'rollout', 'position', 'group')}
    cols['step'] = c * n_steps // n_calls
    # constant turn and rollout: turn lockstep then orders the sub-queries of a step by position, i.e. in split order
    cols['position'] = c
    cols['group'] = np.repeat(np.arange(len(group_ptr) - 1), np.diff(group_ptr)).astype(np.int64)
    return cols, n_calls, np.diff(group_ptr).tolist(), n_steps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('workload', choices=('w1', 'w2', 'w3', 'w4', 'deep'))
    ap.add_argument('--split', required=True, help='split key: val or test (W3 also s1_40, s201_240)')
    ap.add_argument('--sets', required=True, help='per-query sets of the split (.npz with e_ptr, e_nodes, batch_ptr)')
    ap.add_argument('--splits', required=True, help='splits.npz of the workload (<split>_ptr, <split>_q)')
    ap.add_argument('--calls', help='calls.jsonl.gz of the workload (W2, W3, W4)')
    ap.add_argument('--steps', default=None,
                    help='first,last step of the window (W2, W3); default: W3 val 41,70, test 71,200, s1_40 1,40, '
                         's201_240 201,240; W2 val 241,280, test 281,400')
    ap.add_argument('--source', default=None,
                    help='run tag of the evaluated calls in calls.jsonl.gz (default: perturnnorm for W3, olmo3 for W4)')
    ap.add_argument('--drop', default=None,
                    help='W3: drop.json of the workload (prepare/extract_calls.py); its process-run groups are skipped '
                         'as in make_splits.py (default: no group is skipped)')
    ap.add_argument('--train-last-step', type=int, default=240, help='W2: prompts of steps up to this one are excluded')
    ap.add_argument('--strings', help='W4: qstr.json, the distinct query strings')
    ap.add_argument('--questions', help='W4: meta_<split>.npy, the questions of the split')
    ap.add_argument('--block', type=int, default=1024, help='W4: questions per step at one checkpoint')
    ap.add_argument('--per-step', type=int, default=226, help='W1: trajectories per step')
    ap.add_argument('--n-steps', type=int, default=None, help='deep: steps of the split (default: val 30, test 130)')
    ap.add_argument('--out', required=True, help='output .npz')
    a = ap.parse_args()

    q = np.load(a.sets)
    sp = np.load(a.splits)
    string = sp[f'{a.split}_q'].astype(np.int64)   # distinct-string id of every sub-query, in split order
    n_calls = len(string)
    if a.workload == 'w3':
        a.source = a.source or 'perturnnorm'
        lo, hi = [int(x) for x in a.steps.split(',')] if a.steps else W3_WINDOWS[a.split]
        cols, c, sizes, n_steps = w3(a, n_calls, lo, hi)
        assert sizes == np.diff(sp[f'{a.split}_ptr']).tolist(), 'group sizes differ from the splits file'
    elif a.workload == 'w2':
        lo, hi = [int(x) for x in a.steps.split(',')] if a.steps else W2_WINDOWS[a.split]
        cols, c, sizes, n_steps = w2(a, n_calls, lo, hi)
    elif a.workload == 'w4':
        a.source = a.source or 'olmo3'
        cols, c, sizes, n_steps = w4(a, n_calls)
    elif a.workload == 'deep':
        cols, c, sizes, n_steps = deep(a, n_calls, q['batch_ptr'])
    else:
        cols, c, sizes, n_steps = w1(a, n_calls, q['batch_ptr'])
    # the rebuilt order must cover exactly the sub-queries of the split, group by group
    assert c == n_calls == len(q['e_ptr']) - 1, (c, n_calls, len(q['e_ptr']) - 1)
    assert np.array_equal(np.diff(q['batch_ptr']), sizes), 'sub-queries per group differ from the sets file'
    np.savez(a.out, n_steps=np.int64(n_steps), string=string, **cols)
    print(f'{a.workload} {a.split}: {n_calls} sub-queries, {len(sizes)} groups, {n_steps} steps -> {a.out}')


if __name__ == '__main__':
    main()
