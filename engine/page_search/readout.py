"""Page-search readouts: the L of every arm chosen on validation sweeps (select), and the test page reads per call of
HyperPage's page-search layouts against MARGO's at their chosen L, with recall (saving)."""
# [Paper] Section VII-B, MARGO's page-search engine: "Page search takes the smallest L with validation recall@10 of
# 0.95" (Table V caption); "a record cache of the whole budget ranked by training frequency, which beats its own cache
# selection" (select, --rule fewest-reads over W in {1, 2, 4}); "HyperPage's page-search layouts read 11.8%, 8.3% and
# 7.5% fewer pages than MARGO's on every draw, at test recall within 0.3 points" (saving, one MARGO run and one run per
# training draw of HyperPage). Inputs are the logs of search_disk_index (HPSTAT lines) and the outputs of recall.py.
import argparse
import json

import numpy as np


def hpstat(log):
    """HPSTAT lines of a search_disk_index log, one dict per L."""
    out = []
    for line in open(log):
        if line.startswith('HPSTAT'):
            t = line.split(); out.append(dict(zip(t[1::2], t[2::2])))
    return out


def sweep_rows(log, recall_txt):
    """Rows (W, L, reads, sub-queries, recall, QPS) of one sweep."""
    st = {int(kv['L']): kv for kv in hpstat(log)}
    rows = []
    for line in open(recall_txt):
        if not line.strip():
            continue
        L = int(line.split()[1]); r = json.loads(line.split(' ', 2)[2]); kv = st[L]
        rows.append(dict(W=int(kv['W']), L=L, reads=int(kv['sum_ios']), nq=int(kv['nq']), recall=r['recall_at_10'],
                         qps=float(kv['qps'])))
    return rows


def select(a):
    arms = {}
    for name, log, rec in a.sweep:
        arms.setdefault(name, []).extend(sweep_rows(log, rec))
    # smallest-L: the first L, in increasing order, that reaches the target (W = 1 sweeps);
    # fewest-reads: among the (W, L) that reach it, the fewest validation reads, ties to the smaller W, then L
    key = (lambda r: (r['L'], r['W'])) if a.rule == 'smallest-L' else (lambda r: (r['reads'], r['W'], r['L']))
    sel = {}
    for name, rows in arms.items():
        ok = [r for r in rows if r['recall'] >= a.target]
        if not ok:
            sel[name] = None; print(f'{name:24s} no (W, L) reaches recall {a.target}'); continue
        b = dict(min(ok, key=key)); b['reads_per_query'] = b['reads'] / b['nq']; sel[name] = b
        print(f"{name:24s} W={b['W']} L={b['L']:4d} recall={b['recall']:.4f} reads/query={b['reads_per_query']:.1f}")
    out = dict(rule=a.rule, target=a.target, selected=sel, fewer={})
    for spec in a.fewer:
        # of two arms (e.g. MARGO's own cache selection and the training-frequency cache), the one whose selected
        # configuration reads fewer validation pages
        x, y = spec.split(',')
        if sel.get(x) and sel.get(y):
            out['fewer'][spec] = min((x, y), key=lambda t: sel[t]['reads'])
            print(f'fewer validation reads of {x} and {y}: {out["fewer"][spec]}')
    return out


def saving(a):
    def arm(log, rec):
        kv = hpstat(log)[0]; r = json.load(open(rec))
        return dict(L=int(kv['L']), reads_per_call=int(kv['sum_ios']) / int(kv['nq']), recall=r['recall_at_10'])

    m = arm(*a.margo); h = [arm(*x) for x in a.hp]
    rp = [x['reads_per_call'] for x in h]; sv = [100 * (1 - x / m['reads_per_call']) for x in rp]
    out = dict(margo=m, hp=dict(L=[x['L'] for x in h], reads=rp, mean=float(np.mean(rp)),
                                std=float(np.std(rp, ddof=1)) if len(rp) > 1 else 0.0,
                                recall=[x['recall'] for x in h], recall_mean=float(np.mean([x['recall'] for x in h]))),
               saving=dict(per_draw=sv, mean=float(np.mean(sv)), std=float(np.std(sv, ddof=1)) if len(sv) > 1 else 0.0,
                           all_positive=all(x > 0 for x in sv)))
    print(f"MARGO L={m['L']} reads/call {m['reads_per_call']:.2f} recall {m['recall']:.4f}")
    print(f"HyperPage L={out['hp']['L']} reads/call {[round(x, 2) for x in rp]} recall "
          f"{[round(x, 4) for x in out['hp']['recall']]} (mean {out['hp']['recall_mean']:.4f})")
    print(f"saving per draw {[round(x, 2) for x in sv]}  mean {out['saving']['mean']:.2f} +- "
          f"{out['saving']['std']:.2f}  every draw positive: {out['saving']['all_positive']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select', help='L (and W) of every arm from its validation sweeps')
    s.add_argument('--sweep', nargs=3, action='append', required=True, metavar=('ARM', 'LOG', 'RECALL'),
                   help='arm name, sweep log and the output of recall.py sweep; repeat (sweeps of one arm are pooled)')
    s.add_argument('--target', type=float, default=0.95, help='validation recall@10 to reach')
    s.add_argument('--rule', choices=('smallest-L', 'fewest-reads'), default='smallest-L')
    s.add_argument('--fewer', action='append', default=[], metavar='ARM,ARM',
                   help='report which of two arms reads fewer validation pages at its selected configuration')
    s.add_argument('--out', default=None, help='JSON output file')
    v = sub.add_parser('saving', help='test page reads per call and recall of HyperPage against MARGO')
    v.add_argument('--margo', nargs=2, required=True, metavar=('LOG', 'RECALL'),
                   help="log of the test run with MARGO's layout and its recall.py recall output")
    v.add_argument('--hp', nargs=2, action='append', required=True, metavar=('LOG', 'RECALL'),
                   help="the same for HyperPage's page-search layout of one training draw; repeat per draw")
    v.add_argument('--out', default=None, help='JSON output file')
    a = ap.parse_args()
    out = select(a) if a.cmd == 'select' else saving(a)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
