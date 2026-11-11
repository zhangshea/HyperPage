"""Columns of the test-workload table: units, sub-queries, mean |e_q|, mean |E_b|, overlap, L and recall@10, from the
test hypergraph files and the summary of the logging run."""
import argparse
import json
import os

import numpy as np

# [Paper] Table II columns; overlap is 1 - sum_b |E_b| / sum_q |e_q| over the test split (Table II caption)


def row(name, hyper_dir, log_summary, recall_json):
    U = np.load(os.path.join(hyper_dir, 'test_U.npz'))
    Q = np.load(os.path.join(hyper_dir, 'test_Q.npz'))
    s = json.load(open(log_summary))
    recall = json.load(open(recall_json))['recall10'] if recall_json else s['test_sample']['recall10']
    return dict(workload=name, units=int(len(U['e_ptr']) - 1), sub_queries=int(len(Q['e_ptr']) - 1),
                mean_e_q=float(np.diff(Q['e_ptr']).mean()), mean_E_b=float(np.diff(U['e_ptr']).mean()),
                overlap=float(1 - len(U['e_nodes']) / max(1, len(Q['e_nodes']))), L=int(s['L']),
                recall10=float(recall), record_bytes=s['meta']['max_node_len'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--row', nargs='+', action='append', required=True,
                    metavar='ARG', help='NAME HYPER_DIR LOG_SUMMARY [RECALL_JSON]; repeat once per workload. '
                                        'RECALL_JSON (deep_recall.py output) replaces the test recall of the summary')
    ap.add_argument('--out', default=None, help='optional JSON output')
    a = ap.parse_args()
    rows = []
    for r in a.row:
        assert len(r) in (3, 4), r
        rows.append(row(r[0], r[1], r[2], r[3] if len(r) == 4 else None))
    print(f"{'Workload':14s} {'units':>8s} {'sub-queries':>12s} {'|e_q|':>8s} {'|E_b|':>8s} {'overlap':>8s} "
          f"{'L':>6s} {'recall@10':>9s}")
    for r in rows:
        print(f"{r['workload']:14s} {r['units']:8,d} {r['sub_queries']:12,d} {r['mean_e_q']:8.1f} "
              f"{r['mean_E_b']:8,.0f} "
              f"{100 * r['overlap']:7.1f}% {r['L']:6,d} {r['recall10']:9.3f}")
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
