"""Check one-thread runs of the hybrid-cache engine against the replay: per-step page reads equal to the replayed reads,
logged expansions equal to the replayed sets (Assumption 1), and identical results across arms."""
# [Paper] Section VII-A: "With one thread, it reads exactly the replayed pages of every 1M validation step under the
# hybrid cache." Runs: hybrid_search with 1 thread, order frontier, log 1, on the files of stage_hybrid.py.
import argparse
import json

import numpy as np

from hyperpage import formats


def read_res_full(prefix):
    """Result ids and distances of a run (<prefix>.res.bin)."""
    with open(prefix + '.res.bin', 'rb') as f:
        nq, k = np.fromfile(f, np.uint32, 2)
        ids = np.fromfile(f, np.uint32, int(nq) * int(k)).reshape(nq, k)
        d = np.fromfile(f, np.float32, int(nq) * int(k)).reshape(nq, k)
    return ids, d


def read_steps(prefix):
    """Per-step counters of hybrid_search (<prefix>.steps.bin, 14 columns; column 0 = page reads)."""
    a = np.fromfile(prefix + '.steps.bin', np.uint64)
    return a[1:].reshape(int(a[0]), 14)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sets', required=True, help='per-query sets of the split (.npz with e_ptr, e_nodes)')
    ap.add_argument('--lock', required=True, help='engine call order (<key>_lock.npy from stage_hybrid.py)')
    ap.add_argument('--run', nargs=3, action='append', required=True, metavar=('NAME', 'PREFIX', 'REFERENCE'),
                    help='run name, output prefix of the run, replayed per-step reads (.npy); repeat for every run')
    ap.add_argument('--out', default=None, help='JSON output file (default: print only)')
    a = ap.parse_args()

    Q = np.load(a.sets); e_ptr, e_nodes = Q['e_ptr'].astype(np.int64), Q['e_nodes'].astype(np.uint32)
    lock = np.load(a.lock); nq = len(lock)
    lens = e_ptr[lock + 1] - e_ptr[lock]
    exp_ptr = np.zeros(nq + 1, np.int64); exp_ptr[1:] = np.cumsum(lens)
    idx = np.repeat(e_ptr[lock] - exp_ptr[:-1], lens) + np.arange(exp_ptr[-1])
    expected = e_nodes[idx]

    def sequences(prefix):
        """Calls whose logged expanded sequence differs from the replayed set in expansion order."""
        noff, nodes = formats.read_expanded_sets(prefix)
        assert len(noff) == nq + 1
        if not np.array_equal(noff, exp_ptr):
            return dict(calls_differing=int(np.sum(np.diff(noff) != lens)), lengths_equal=False)
        diff = nodes.astype(np.uint32) != expected
        bad = int(np.unique(np.searchsorted(exp_ptr, np.flatnonzero(diff), side='right') - 1).size) if diff.any() else 0
        return dict(calls_differing=bad, lengths_equal=True, nodes_compared=int(len(nodes)))

    out = dict(calls=int(nq), runs={})
    ids0 = d0 = None
    for name, prefix, ref in a.run:
        st = read_steps(prefix); ref = np.load(ref).astype(np.int64); eng = st[:, 0].astype(np.int64)
        ids, d = read_res_full(prefix)
        if ids0 is None:
            ids0, d0 = ids, d
        out['runs'][name] = dict(reads=int(eng.sum()), replay_reads=int(ref.sum()), steps=int(len(ref)),
                                 steps_equal=int(np.sum(eng == ref)) if len(eng) == len(ref) else -1,
                                 every_step_equal=bool(np.array_equal(eng, ref)),
                                 relative_difference=abs(int(eng.sum()) - int(ref.sum())) / max(1, int(ref.sum())),
                                 expanded=int(st[:, 4].sum()), cached=int(st[:, 5].sum()),
                                 ids_equal_first_run=bool(np.array_equal(ids, ids0)),
                                 dists_equal_first_run=bool(np.array_equal(d, d0)), a1=sequences(prefix))
    out['all_pass'] = all(r['every_step_equal'] and r['a1']['calls_differing'] == 0 and r['ids_equal_first_run']
                          and r['dists_equal_first_run'] for r in out['runs'].values())
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
