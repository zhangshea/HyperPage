"""Read-speed check before a timed round: the files of the round must read at nearly the same speed, and each file near
its speed at the first check of the run. Exit status 0 lets the round run; 1 makes it void (no timed runs)."""
# [Paper] Section VII-B, Search time: a pair is void when a check fixed before the run flags the read speed of the two
# files. Rule: (max - min) / min of the per-file mean latencies <= --max-gap, and every file within --max-drift of its
# reference, the per-file means of the first check (written to --reference when that file does not exist yet).
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('speeds', help='output of read_speed.py for the files of the round')
    ap.add_argument('--reference', required=True, help='JSON of per-file reference means; created at the first check')
    ap.add_argument('--max-gap', type=float, default=0.05, help='largest relative gap between the files of the round')
    ap.add_argument('--max-drift', type=float, default=0.10,
                    help='largest relative change of a file from its reference')
    ap.add_argument('--drift-form', choices=('current_over_reference', 'reference_over_current'),
                    default='current_over_reference',
                    help='drift = |current / reference - 1| or |reference / current - 1|')
    ap.add_argument('--cross', action='store_true',
                    help='admit a round that has its own reference, on the same files: every mean of the round '
                         'within --max-drift of every reference mean (all pairs, no gap check); the reference '
                         'must exist')
    a = ap.parse_args()
    cur = {f: v['mean_us'] for f, v in json.load(open(a.speeds)).items()}
    if a.cross:
        # [Paper] Table V caption, medians of 4 valid rounds at 1M, 5%: a valid round with a reference of its own enters
        # the cell when its read speeds match the reference of the other rounds
        ref = json.load(open(a.reference))
        if a.drift_form == 'current_over_reference':
            drift = max(abs(x / g - 1) for x in cur.values() for g in ref.values())
        else:
            drift = max(abs(g / x - 1) for x in cur.values() for g in ref.values())
        ok = drift <= a.max_drift
        print(json.dumps(dict(means=cur, reference=ref, drift=drift, pass_=ok)))
        sys.exit(0 if ok else 1)
    if not os.path.exists(a.reference):
        json.dump(cur, open(a.reference, 'w'), indent=1)
    ref = json.load(open(a.reference))
    missing = sorted(f for f in cur if f not in ref)
    if missing:
        sys.exit('no reference speed for: ' + ', '.join(missing) + ' (the reference must list the files of the round)')
    m = list(cur.values()); gap = (max(m) - min(m)) / min(m)
    if a.drift_form == 'current_over_reference':
        drift = max(abs(x / ref[f] - 1) for f, x in cur.items())
    else:
        drift = max(abs(ref[f] / x - 1) for f, x in cur.items())
    ok = gap <= a.max_gap and drift <= a.max_drift
    print(json.dumps(dict(means=cur, gap=gap, drift=drift, pass_=ok)))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
