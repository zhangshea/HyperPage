"""Choice of the candidate-list size L by validation recall@10, test recall@10 at that L, and the logging run that
traverses every needed sub-query string once with the stock beam search and the traversal logger (W = 4, K = 10)."""
import argparse
import json
import os
import subprocess

import numpy as np

from hyperpage import formats

# [Paper] Section VII-A, Metrics: the smallest L with validation recall@10 of at least 0.95 (L = 500 at 1M, 800 at 21M)
L_GRID = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 230, 260, 300, 350,
          400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1700, 2000, 2500, 3000]


def run(a, qfile, L, out):
    """One run of the logging search application on a query file; returns the counters of its summary line."""
    cmd = [a.app, a.data_type, 'l2', a.index_prefix, qfile, str(L), str(a.beam_width), '10', str(a.threads), out, '0']
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS=str(a.threads)))
    line = [ln for ln in r.stdout.splitlines() if ln.startswith('nq ')]
    if r.returncode != 0 or not line:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit('search failed')
    t = line[0].split()
    return {t[i]: float(t[i + 1]) for i in range(0, len(t) - 1, 2)}


def load_queries(path, raw_dim):
    """Query vectors indexed by string id: an .npy matrix, or a headerless float32 file with raw_dim columns."""
    if raw_dim:
        return np.memmap(path, dtype=np.float32, mode='r').reshape(-1, raw_dim)
    return np.load(path, mmap_mode='r')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--app', required=True, help='path of the logging search application of engine/')
    ap.add_argument('--data-type', choices=['int8', 'float'], default='int8')
    ap.add_argument('--index-prefix', required=True)
    ap.add_argument('--data-dir', required=True,
                    help='gtq_val.bin, gt_val.npy, gtq_test.bin, gt_test.npy, queries_needed.npy, qx8.npy')
    ap.add_argument('--out-dir', required=True, help='LOG_DIR/<workload>')
    ap.add_argument('--fixed-L', type=int, default=None, help='skip the validation sweep and use this L')
    ap.add_argument('--target-recall', type=float, default=0.95)
    ap.add_argument('--beam-width', type=int, default=4)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--skip-test', action='store_true', help='no test recall (DEEP1M: see deep_recall.py)')
    ap.add_argument('--queries', default=None, help='query vectors (default: <data-dir>/qx8.npy)')
    ap.add_argument('--raw-dim', type=int, default=0, help='--queries is headerless float32 with this many columns')
    ap.add_argument('--chunk-size', type=int, default=400000,
                    help='strings per logging run all_c<k>; 0 logs all needed strings in one run')
    a = ap.parse_args()
    D, OUT = a.data_dir, a.out_dir
    os.makedirs(OUT, exist_ok=True)
    meta = formats.read_index_meta(a.index_prefix + '_disk.index')
    print('meta', meta, flush=True)

    gtv = np.load(os.path.join(D, 'gt_val.npy'))
    sweep = []
    chosen = a.fixed_L
    for L in ([] if chosen else L_GRID):
        st = run(a, os.path.join(D, 'gtq_val.bin'), L, f'{OUT}/sweep_val_L{L}')
        st['recall10'] = formats.recall_at(formats.read_results(f'{OUT}/sweep_val_L{L}'), gtv, 10)
        st['L'] = L
        sweep.append(st)
        print('sweep', L, st['recall10'], st['mean_expanded'], flush=True)
        for ext in ('.log.bin', '.res.bin'):
            os.remove(f'{OUT}/sweep_val_L{L}{ext}')
        if st['recall10'] >= a.target_recall:
            chosen = L
            break
    if chosen is None:
        best = max(sweep, key=lambda s: s['recall10'])
        raise SystemExit(f'no L of the grid reaches validation recall@10 {a.target_recall} '
                         f'(best: {best["recall10"]:.4f} at L = {best["L"]}); pass a fixed L with --fixed-L')
    res = dict(meta=meta, W=a.beam_width, K=10, cache_nodes=0, L=chosen, sweep=sweep)
    if not a.skip_test:
        st = run(a, os.path.join(D, 'gtq_test.bin'), chosen, f'{OUT}/gt_test')
        gtt = np.load(os.path.join(D, 'gt_test.npy'))
        st['recall10'] = formats.recall_at(formats.read_results(f'{OUT}/gt_test'), gtt, 10)
        res['test_sample'] = st

    # Logging run: the needed strings in ascending id order, split into query files of chunk_size strings.
    # The expanded sets do not depend on the layout (A1), so this one run serves every layout.
    need = np.load(os.path.join(D, 'queries_needed.npy'))
    Q = load_queries(a.queries or os.path.join(D, 'qx8.npy'), a.raw_dim)
    CH = a.chunk_size or len(need)
    chunks = []
    for c in range(0, len(need), CH):
        q = np.ascontiguousarray(Q[need[c:c + CH]])
        qf = f'{OUT}/logq_c{c // CH}.bin'
        with open(qf, 'wb') as f:
            np.array([len(q), q.shape[1]], np.int32).tofile(f)
            q.tofile(f)
        st = run(a, qf, chosen, f'{OUT}/all_c{c // CH}')
        chunks.append(st)
        os.remove(qf)
        print('logged chunk', c // CH, st, flush=True)
    res['log_chunks'] = chunks
    res['chunk_size'] = CH
    res['n_logged'] = int(len(need))
    json.dump(res, open(f'{OUT}/tune_log_summary.json', 'w'), indent=1)
    print('DONE L', chosen, 'test recall', res.get('test_sample', {}).get('recall10'), flush=True)


if __name__ == '__main__':
    main()
