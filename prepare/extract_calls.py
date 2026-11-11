"""Extraction of the search sub-queries of W1-W4 from the public trajectories and rollouts into calls.jsonl.gz.

One JSON row per sub-query, in file order; the row layout per workload is given in prepare/README.md."""
import argparse
import ast
import collections
import glob
import gzip
import hashlib
import json
import os
import re

TOOL_CALL = re.compile(r'<tool_call>(.*?)</tool_call>', re.S)
QUESTION = re.compile(r'Question:\s*(.*?)\s*\nassistant\s*\n?$', re.S)
OLMO_CALL = re.compile(r'<function_calls>\s*search\(\s*query_list\s*=\s*(\[.*?\])\s*\)\s*</function_calls>', re.S)
QUERY_FIELD = re.compile(r'"query"\s*:\s*"((?:[^"\\]|\\.)*)"')


def norm(q):
    """Prompt identity: lower case, collapsed whitespace, trailing '?' removed."""
    q = ' '.join(q.lower().split())
    return q[:-1].rstrip() if q.endswith('?') else q


def prompt_key(q):
    return hashlib.sha1(norm(q).encode()).hexdigest()[:16]


def rollout_calls(output):
    """Query lists of the parseable <tool_call> blocks of one rollout; empty or non-string queries are skipped."""
    res, bad = [], 0
    for t in TOOL_CALL.findall(output):
        try:
            a = json.loads(t.strip()).get('arguments', {})
            ql = a.get('query_list', [a['query']] if 'query' in a else [])
        except Exception:
            bad += 1
            continue
        res.append([q for q in ql if isinstance(q, str) and q.strip()])
    return res, bad


def w1(src, out):
    """W1: browser.search calls of the OpenResearcher trajectories with a parseable, non-empty query.

    call_idx counts every browser.search message of a trajectory, including the unparseable ones."""
    import pyarrow.parquet as pq
    kept = drop = 0
    with gzip.open(os.path.join(out, 'calls.jsonl.gz'), 'wt', encoding='utf-8') as g:
        for f in sorted(glob.glob(os.path.join(src, 'seed_*', '*.parquet'))):
            seed = int(f.split('seed_')[1].split('/')[0])
            t = pq.read_table(f, columns=['qid', 'status', 'messages'])
            for qid, status, msgs in zip(t.column('qid').to_pylist(), t.column('status').to_pylist(),
                                         t.column('messages').to_pylist()):
                ns = 0
                for m in msgs or []:
                    if (m.get('recipient') or '') != 'browser.search':
                        continue
                    c = m.get('content')
                    txt = c[0].get('text') if isinstance(c, list) and c else (c or '')
                    q = None
                    try:
                        q = json.loads(txt).get('query')
                    except Exception:
                        # message text that is not valid JSON: take the "query" field by pattern
                        mm = QUERY_FIELD.search(txt or '')
                        if mm:
                            try:
                                q = json.loads('"' + mm.group(1) + '"')
                            except Exception:
                                q = mm.group(1)
                    if q is None or not str(q).strip():
                        drop += 1
                    else:
                        g.write(json.dumps([seed, int(qid), status, ns, str(q)], ensure_ascii=False) + '\n')
                        kept += 1
                    ns += 1
            print(f, t.num_rows, flush=True)
    print('W1 calls kept', kept, 'dropped', drop)


def w2(src, out, steps):
    """W2: every element of every parseable search call's query_list in the outcome-run rollouts."""
    kept = fail = empty = 0
    with gzip.open(os.path.join(out, 'calls.jsonl.gz'), 'wt', encoding='utf-8') as g:
        for s in range(1, steps + 1):
            for li, line in enumerate(open(os.path.join(src, f'{s}.jsonl'), encoding='utf-8')):
                x = json.loads(line)
                ph = hashlib.sha1(x['input'].encode()).hexdigest()[:16]
                for ti, t in enumerate(TOOL_CALL.findall(x['output'])):
                    try:
                        a = json.loads(t.strip()).get('arguments', {})
                        ql = a.get('query_list', [a['query']] if 'query' in a else [])
                    except Exception:
                        fail += 1
                        continue
                    for qi, q in enumerate(ql):
                        if not isinstance(q, str) or not q.strip():
                            empty += 1
                            continue
                        g.write(json.dumps([s, ph, li, ti, qi, q], ensure_ascii=False) + '\n')
                        kept += 1
    print('W2 sub-queries kept', kept, 'unparseable calls', fail, 'empty', empty)


def scan_run(d, steps, keep_steps, tag, writer, stats):
    """Prompt counts of every step of one run; writes the sub-queries of the steps in keep_steps."""
    per_step = {}
    for s in steps:
        qs = collections.Counter()
        n = 0
        for li, line in enumerate(open(os.path.join(d, f'{s}.jsonl'), encoding='utf-8')):
            x = json.loads(line)
            n += 1
            m = QUESTION.search(x['input'])
            q = m.group(1) if m else None
            if q is None:
                stats['no_question'] += 1
                continue
            k = prompt_key(q)
            qs[k] += 1
            if s in keep_steps:
                calls, bad = rollout_calls(x['output'])
                stats['bad_calls'] += bad
                for ti, ql in enumerate(calls):
                    for qi, sq in enumerate(ql):
                        writer.write(json.dumps([tag, s, k, li, ti, qi, sq], ensure_ascii=False) + '\n')
                        stats['subqueries'] += 1
        per_step[s] = dict(lines=n, prompts=qs)
    return per_step


def w3(outcome, process, out, steps, a):
    """W3: process-run steps 1..train_lo-1 and outcome-run steps train_lo..steps, with the prompt checks of
    Section VII-A, Datasets (prompt sets per step, held-out prompts that occur in a training step)."""
    stats = collections.Counter()
    with gzip.open(os.path.join(out, 'calls.jsonl.gz'), 'wt', encoding='utf-8') as g:
        pt = scan_run(process, range(1, steps + 1), set(range(1, a.train_lo)), 'perturnnorm', g, stats)
        st_p = dict(stats)
        stats.clear()
        fg = scan_run(outcome, range(1, steps + 1), set(range(a.train_lo, steps + 1)), '4gpu', g, stats)
        st_g = dict(stats)
    chk = {'process_run_extract': st_p, 'outcome_run_extract': st_g}
    chk['process_run_lines_per_step'] = dict(collections.Counter(v['lines'] for v in pt.values()))
    chk['process_run_prompts_per_step'] = dict(collections.Counter(len(v['prompts']) for v in pt.values()))
    chk['process_run_rollouts_per_prompt'] = dict(collections.Counter(c for v in pt.values()
                                                                      for c in v['prompts'].values()))
    chk['outcome_run_lines_per_step'] = dict(collections.Counter(v['lines'] for v in fg.values()))
    chk['outcome_run_rollouts_per_prompt'] = dict(collections.Counter(c for v in fg.values()
                                                                      for c in v['prompts'].values()))
    # [Paper] Section VII-A: prompt sets align between the two runs on all steps
    al = [s for s in range(1, steps + 1) if set(pt[s]['prompts']) == set(fg[s]['prompts'])]
    chk['steps_with_identical_prompt_sets'] = len(al)
    chk['steps_not_aligned'] = [s for s in range(1, steps + 1) if s not in set(al)]
    occ = collections.Counter(k for v in pt.values() for k in v['prompts'])
    chk['process_run_prompts_in_more_than_one_step'] = sum(1 for c in occ.values() if c > 1)
    chk['process_run_distinct_prompts'] = len(occ)
    # [Paper] Section VII-A: no validation or test prompt occurs in a training step; such groups would be dropped
    train = {k for s in range(a.train_lo, steps + 1) for k in fg[s]['prompts']}
    held = {(s, k) for s in range(a.heldout_lo, a.heldout_hi + 1) for k in pt[s]['prompts']}
    drop = sorted([s, k] for s, k in held if k in train)
    chk['heldout_prompts_in_training_steps'] = drop
    windows = {(s, k) for s in list(range(1, a.heldout_lo)) + list(range(a.heldout_hi + 1, a.train_lo))
               for k in pt[s]['prompts']}
    chk['window_prompts_in_training_steps'] = sorted([s, k] for s, k in windows if k in train)
    json.dump(chk, open(os.path.join(out, 'checks.json'), 'w'), indent=1)
    json.dump({'drop_groups': drop + chk['window_prompts_in_training_steps']},
              open(os.path.join(out, 'drop.json'), 'w'))
    print(json.dumps({k: v for k, v in chk.items() if 'prompts_in' not in k}, indent=1))
    print('held-out groups to drop:', len(drop), '; window groups to drop:',
          len(chk['window_prompts_in_training_steps']))


def w4(outcome, olmo, out, steps, checkpoints):
    """W4: all outcome-run steps (training) and the OLMo-3 search calls per (question, checkpoint); questions that are
    outcome-run prompts are listed in drop.json."""
    stats = collections.Counter()
    w2prompts = set()
    q_of = {}
    per = collections.Counter()
    bad = 0
    with gzip.open(os.path.join(out, 'calls.jsonl.gz'), 'wt', encoding='utf-8') as g:
        fg = scan_run(outcome, range(1, steps + 1), set(range(1, steps + 1)), '4gpu', g, stats)
        for v in fg.values():
            w2prompts |= set(v['prompts'])
        for ck in checkpoints:
            for line in open(os.path.join(olmo, f'eval_step{ck}_olmo3_metrics_samples.jsonl'), encoding='utf-8'):
                x = json.loads(line)
                idx = int(x['index'])
                k = prompt_key(x['question'])
                if q_of.setdefault(idx, k) != k:
                    stats['index_question_mismatch'] += 1
                per[idx] += 1
                ti = 0
                for turn in x['conversation']:
                    if turn.get('role') != 'assistant':
                        continue
                    for lst in OLMO_CALL.findall(turn.get('content') or ''):
                        try:
                            ql = json.loads(lst)
                        except Exception:
                            try:
                                ql = ast.literal_eval(lst)
                            except Exception:
                                bad += 1
                                continue
                        for qi, sq in enumerate(ql):
                            if isinstance(sq, str) and sq.strip():
                                g.write(json.dumps(['olmo3', ck, k, idx, ti, qi, sq], ensure_ascii=False) + '\n')
                                stats['olmo3_subqueries'] += 1
                        ti += 1
    drop = sorted(i for i, k in q_of.items() if k in w2prompts)
    chk = dict(stats=dict(stats), olmo3_bad_calls=bad, questions=len(q_of),
               trajectories_per_question=dict(collections.Counter(per.values())),
               questions_that_are_outcome_run_prompts=len(drop))
    json.dump(chk, open(os.path.join(out, 'checks.json'), 'w'), indent=1)
    json.dump({'drop_index': drop}, open(os.path.join(out, 'drop.json'), 'w'))
    print(json.dumps(chk, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('workload', choices=['w1', 'w2', 'w3', 'w4'])
    ap.add_argument('--out', required=True, help='output directory (DATA_DIR/<workload>)')
    ap.add_argument('--trajectories', help='W1: directory with seed_*/train-*.parquet of OpenResearcher-Dataset')
    ap.add_argument('--outcome-run', help='W2, W3, W4: directory with <step>.jsonl of the outcome-reward run')
    ap.add_argument('--process-run', help='W3: directory with <step>.jsonl of the process-reward run')
    ap.add_argument('--olmo', help='W4: directory with eval_step<ckpt>_olmo3_metrics_samples.jsonl')
    ap.add_argument('--steps', type=int, default=400, help='steps per run')
    ap.add_argument('--train-lo', type=int, default=241, help='W3: first outcome-run training step')
    ap.add_argument('--heldout-lo', type=int, default=41, help='W3: first validation step of the process run')
    ap.add_argument('--heldout-hi', type=int, default=200, help='W3: last test step of the process run')
    ap.add_argument('--checkpoints', default='330,340,350,360,370,380,390,400', help='W4: OLMo-3 checkpoints')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.workload == 'w1':
        w1(a.trajectories, a.out)
    elif a.workload == 'w2':
        w2(a.outcome_run, a.out, a.steps)
    elif a.workload == 'w3':
        w3(a.outcome_run, a.process_run, a.out, a.steps, a)
    else:
        w4(a.outcome_run, a.olmo, a.out, a.steps, [int(c) for c in a.checkpoints.split(',')])


if __name__ == '__main__':
    main()
