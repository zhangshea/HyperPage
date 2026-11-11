"""Recall@10 of page-search runs against exact neighbours (prepare/ground_truth.py): the staged sub-queries that have
ground truth (subset), and the recall of one result file (recall) or of every L of a sweep (sweep)."""
# [Paper] Section VII-B, MARGO's page-search engine, and Table V caption: page search takes the smallest L with
# validation recall@10 of 0.95; "Test recall@10 is 0.946 to 0.954". Two forms of ground truth are used:
#   mapped  (1M): rows of a staged query file in turn lockstep (<key>_q.bin of checks/stage_hybrid.py); a row has ground
#                 truth when its query string is among the sampled strings (gtq_<key>_strid.npy), and recall is the
#                 mean over those rows, so a string counts once per sub-query that issues it;
#   aligned (full corpus): the sampled queries themselves (gtq_<key>.bin), result row i against ground-truth row i.
# Result files are those of MARGO's search_disk_index: <result prefix>_<L>_idx_uint32.bin (u32 rows, u32 K, ids).
import argparse
import json
import re

import numpy as np


def read_ids(path):
    h = np.fromfile(path, np.int32, count=2); n, k = int(h[0]), int(h[1])
    return np.fromfile(path, np.uint32, offset=8, count=n * k).reshape(n, k)


def string_rows(a):
    """Query string of every staged row (lockstep order) and the ground-truth row of every sampled string."""
    strid = np.load(a.schedule)['string'].astype(np.int64)[np.load(a.lock)]
    row = {int(s): i for i, s in enumerate(np.load(a.gt_strings))}
    return strid, row


def recall(a, path):
    """Mean recall@10 over the result rows that have ground truth."""
    gt = np.load(a.gt); ids = read_ids(path)
    if a.schedule is None:   # aligned rows
        assert len(ids) == len(gt), (len(ids), len(gt))
        rec = [len(set(x[:10].tolist()) & set(y[:10].tolist())) / 10 for x, y in zip(ids, gt)]
        return dict(recall_at_10=float(np.mean(rec)), covered=len(rec), rows=len(ids))
    strid, row = string_rows(a)
    rr = np.arange(len(strid)) if a.rows is None else np.load(a.rows)
    assert len(rr) == len(ids), (len(rr), len(ids))
    rec = [len(set(ids[j][:10].tolist()) & set(gt[row[int(strid[i])]][:10].tolist())) / 10
           for j, i in enumerate(rr) if int(strid[i]) in row]
    return dict(recall_at_10=float(np.mean(rec)), covered=len(rec), rows=len(ids))


def ground_truth_args(p):
    p.add_argument('--gt', required=True, help='exact top-10 of the sampled queries (gt_<key>.npy)')
    p.add_argument('--schedule', default=None,
                   help='mapped form: schedule of the split (replay/make_schedule.py), for the query strings')
    p.add_argument('--lock', default=None, help='mapped form: <key>_lock.npy of checks/stage_hybrid.py')
    p.add_argument('--gt-strings', default=None, help='mapped form: gtq_<key>_strid.npy')
    p.add_argument('--rows', default=None,
                   help='mapped form: staged rows the results cover (<prefix>_rows.npy of subset; default all rows)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('subset', help='query file of the staged rows whose string has ground truth')
    s.add_argument('--queries', required=True, help='staged query file in lockstep order (<key>_q.bin)')
    s.add_argument('--schedule', required=True, help='schedule of the split (replay/make_schedule.py)')
    s.add_argument('--lock', required=True, help='<key>_lock.npy of checks/stage_hybrid.py')
    s.add_argument('--gt-strings', required=True, help='gtq_<key>_strid.npy')
    s.add_argument('--out', required=True, help='output prefix: <out>_q.bin and <out>_rows.npy')
    r = sub.add_parser('recall', help='recall@10 of one result file')
    r.add_argument('results', help='<result prefix>_<L>_idx_uint32.bin')
    ground_truth_args(r)
    r.add_argument('--out', default=None, help='JSON output file')
    w = sub.add_parser('sweep', help='recall@10 of every L of a sweep; one line "L <L> <json>" per L')
    w.add_argument('log', help='output of search_disk_index (one HPSTAT line per L)')
    w.add_argument('result_prefix', help='--result_path of the sweep')
    ground_truth_args(w)
    w.add_argument('--out', default=None, help='output text file (default: print only)')
    a = ap.parse_args()

    if a.cmd == 'subset':
        strid, row = string_rows(a)
        have = np.array([i for i in range(len(strid)) if int(strid[i]) in row], np.int64)
        h = np.fromfile(a.queries, np.int32, count=2); nq, d = int(h[0]), int(h[1]); assert nq == len(strid)
        X = np.fromfile(a.queries, np.int8, offset=8).reshape(nq, d)[have]
        with open(f'{a.out}_q.bin', 'wb') as f:
            np.array([len(have), d], np.int32).tofile(f); X.tofile(f)
        np.save(f'{a.out}_rows.npy', have)
        print(json.dumps(dict(calls=nq, covered=len(have), dim=d)))
    elif a.cmd == 'recall':
        res = recall(a, a.results)
        if a.out:
            json.dump(res, open(a.out, 'w'))
        print(json.dumps(res))
    else:
        Ls = [int(m.group(1)) for m in re.finditer(r'^HPSTAT L (\d+) ', open(a.log).read(), re.M)]
        lines = [f'L {L} {json.dumps(recall(a, f"{a.result_prefix}_{L}_idx_uint32.bin"))}' for L in Ls]
        if a.out:
            open(a.out, 'w').write('\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
