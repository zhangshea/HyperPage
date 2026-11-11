"""Validity and statistics of timed rounds: per-run checks (host load, CPU affinity, device drift), void rounds, and per
compared pair of arms the paired time savings, their median, the sign test, seconds per step, reads saved, latency."""
# [Paper] Table IV (pairs, s per step of both arms, reads saved %, time saved % = median of the paired savings), the
# Search-time paragraph (p50 and p99 falls, "faster in every valid pair", sign-test p), and Table V (page reads per call
# and s per step of the page-search and hybrid arms; page search over hybrid time ratios; 64-256 thread runs).
# A round holds one run per arm (two arms for a pair, five for the five-arm rounds); it is void when its read-speed
# check failed (--void-round) or when any of its runs is flagged:
#   host load : over the run's timed section, the promised-idle CPUs (--idle-cpus) carry more than --max-idle-busy busy
#               cores, or the timed disk (--disk) moves more than --max-foreign of the run's own reads in foreign reads
#               and writes (page-search runs: writes only, since their own reads include loading the index and cache);
#   affinity  : affinity_check.py flagged a foreign process on the run's CPUs (or its record is missing);
#   drift     : the I/O time per read of the last fifth of the run differs from the first fifth by more than --max-drift
#               (hybrid: the per-step record of the engine; page search: the 5 s device samples, first 30 s dropped).
# A round in which a run has no summary line (no `summary` or `HPSTAT` line in its log) is reported as incomplete: it
# is counted apart and is neither valid nor void. Statistics use the first --max-pairs valid rounds in the order given.
import argparse
import json
import math
import os
import re

import numpy as np


def cpu_set(s):
    out = set()
    for part in s.split(','):
        a, _, b = part.partition('-'); out.update(range(int(a), int(b or a) + 1))
    return out


def host(p, idle_cpus, disk):
    """Busy cores on the idle CPUs, and bytes read and written on the disk, between the two snapshots of <run>.host."""
    s = open(p).read()

    def sect(n):
        a = s.index('@@' + n + '\n') + len(n) + 3; b = s.find('@@', a); return s[a:b if b >= 0 else None]

    def cpu(t):
        d = {}
        for line in t.splitlines():
            if line.startswith('cpu') and line[3].isdigit():
                q = line.split(); v = list(map(int, q[1:9])); d[int(q[0][3:])] = (sum(v) - v[3] - v[4], sum(v))
        return d

    def dsk(t):
        for line in t.splitlines():
            q = line.split()
            if len(q) > 9 and q[2] == disk:
                return int(q[5]) * 512, int(q[9]) * 512
        raise KeyError(disk)

    c0, c1 = cpu(sect('stat0')), cpu(sect('stat1')); (r0, w0), (r1, w1) = dsk(sect('disk0')), dsk(sect('disk1'))
    busy = {c: (c1[c][0] - c0[c][0]) / max(1, c1[c][1] - c0[c][1]) for c in c0}
    return sum(busy[c] for c in idle_cpus if c in busy), r1 - r0, w1 - w0


def kv(p, key):
    if not os.path.exists(p):
        return None   # a run that left no log counts as incomplete
    m = re.search(rf'^{key} (.*)$', open(p).read(), re.M)
    if not m:
        return None
    a = m.group(1).split(); return dict(zip(a[0::2], a[1::2]))


def fifths_samples(p, disk):
    """I/O time per read (us) in the first and last fifth of a page-search run, from its 5 s device samples."""
    t, ios, ticks = [], [], []
    for line in open(p):
        q = line.split()
        if len(q) > 9 and q[0] == '@@t' and q[4] == disk:
            t.append(float(q[1])); ios.append(int(q[5])); ticks.append(int(q[8]))
    t, ios, ticks = np.array(t), np.array(ios), np.array(ticks)
    keep = t >= t[0] + 30; t, ios, ticks = t[keep], ios[keep], ticks[keep]
    if len(t) < 11:
        return None
    di, dt = np.diff(ios), np.diff(ticks); mid = (t[1:] + t[:-1]) / 2; edges = np.linspace(t[0], t[-1], 6)
    first, last = (mid < edges[1]), (mid >= edges[4])
    return float(dt[first].sum() / max(1, di[first].sum()) * 1e3), float(dt[last].sum() / max(1, di[last].sum()) * 1e3)


def one(prefix, a):
    """Statistics and flags of one run; None when its log has no summary line."""
    why = []
    s = kv(prefix + '.log', 'summary')
    if s is not None:
        r = dict(engine='hybrid', secs=float(s['secs']), calls=int(s['calls']), reads=int(s['page_reads']),
                 s_per_step=float(s['ms_per_step']) / 1e3, p50_ms=float(s['lat_ms_median']),
                 p99_ms=float(s['lat_ms_p99']), peak_mb=int(s['hwm_kb']) / 1024, qps=float(s['calls_per_s']))
        idle, dr, dw = host(prefix + '.host', a.idle, a.disk); own = int(s['proc_read_bytes'])
        if idle > a.max_idle_busy or (dr - own) + dw > a.max_foreign * own:
            why.append(f'host idle {idle:.2f} foreign {((dr - own) + dw) / max(1, own):.3f}')
        x = np.fromfile(prefix + '.steps.bin', np.uint64)
        st = x[1:].reshape(int(x[0]), 14).astype(float); q = len(st) // 5
        f0, f4 = st[:q, 7].sum() / st[:q, 0].sum() / 1e3, st[-q:, 7].sum() / st[-q:, 0].sum() / 1e3
    else:
        s = kv(prefix + '.log', 'HPSTAT')
        if s is None:
            return None
        r = dict(engine='page-search', secs=float(s['secs']), calls=int(s['nq']), reads=int(s['sum_ios']),
                 s_per_step=float(s['secs']) / a.steps, p50_ms=float(s['p50_us']) / 1e3,
                 p99_ms=float(s['p99_us']) / 1e3,
                 peak_mb=float(s['peak_rss_mb']), qps=float(s['qps']))
        idle, dr, dw = host(prefix + '.host', a.idle, a.disk); own = r['reads'] * a.page_size
        if idle > a.max_idle_busy or dw > a.max_foreign * own:
            why.append(f'host idle {idle:.2f} writes {dw / max(1, own):.3f}')
        ff = fifths_samples(prefix + '.dsamp', a.disk); f0, f4 = ff if ff else (1.0, 1.0)
        if ff is None:
            why.append('device samples too short')
    if not (f0 > 0 and f4 > 0):
        why.append('no reads on the timed disk')
    elif abs(f4 / f0 - 1) > a.max_drift:
        why.append(f'fifths {f0:.1f} -> {f4:.1f} us/read')
    try:
        aff = json.load(open(prefix + '.affinity.json'))
        if aff['flag']:
            why.append('affinity: ' + ','.join(o['comm'] for o in aff['offenders']))
    except (OSError, ValueError):
        why.append('affinity missing')
    r.update(fifths_us=[f0, f4], flags=why, reads_per_call=r['reads'] / r['calls'])
    return r


def sign_test(sv):
    """Two-sided exact sign test of the paired savings; zero savings count for neither side, so the sample size is the
    number of non-zero savings (p = 1 when there is none)."""
    pos, neg = int((sv > 0).sum()), int((sv < 0).sum()); n = pos + neg
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * sum(math.comb(n, k) for k in range(min(pos, neg) + 1)) / 2 ** n))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--round', nargs='+', action='append', default=[], metavar='ITEM',
                    help='NAME ARM=PREFIX [ARM=PREFIX ...]: the runs of one round (output prefixes of run_arm.sh)')
    ap.add_argument('--void-round', action='append', default=[], metavar='NAME',
                    help='round whose read-speed check failed (no runs); counts as void')
    ap.add_argument('--run', action='append', default=[], metavar='NAME=PREFIX',
                    help='single run outside rounds (e.g. a throughput run), reported with its flags')
    ap.add_argument('--compare', action='append', default=[], metavar='A:B',
                    help='pair of arms; saving = 1 - t_A / t_B (default HP:MARGO)')
    ap.add_argument('--idle-cpus', required=True, help='CPUs promised idle during the runs (not the runs\' own CPUs)')
    ap.add_argument('--disk', required=True, help='block device of the index files, as named in /proc/diskstats')
    ap.add_argument('--max-idle-busy', type=float, default=0.6, help='busy cores allowed on the idle CPUs')
    ap.add_argument('--max-foreign', type=float, default=0.05, help='foreign disk traffic allowed, share of own reads')
    ap.add_argument('--max-drift', type=float, default=0.15, help='largest change of I/O time per read, first to last'
                                                                  ' fifth')
    ap.add_argument('--max-pairs', type=int, default=10, help='valid rounds used for the statistics')
    ap.add_argument('--steps', type=int, default=130, help='test steps per run (seconds per step of page-search runs)')
    ap.add_argument('--page-size', type=int, default=8192)
    ap.add_argument('--out', default=None, help='JSON output file')
    a = ap.parse_args()
    a.idle = cpu_set(a.idle_cpus)
    compares = [c.split(':') for c in (a.compare or ['HP:MARGO'])]

    rounds, incomplete = [], []
    for item in a.round:
        name, runs = item[0], dict(x.split('=', 1) for x in item[1:])
        st = {arm: one(p, a) for arm, p in runs.items()}
        if any(v is None for v in st.values()):
            # a run without a summary line: the round is incomplete, neither valid nor void, and enters no statistic
            incomplete.append(dict(round=name, missing=[arm for arm, v in st.items() if v is None]))
            continue
        why = [f'{arm}: {w}' for arm, v in st.items() for w in v['flags']]
        rounds.append(dict(round=name, void=bool(why), why=why, runs=st))
    rounds += [dict(round=n, void=True, why=['read-speed check']) for n in a.void_round]
    ok = [x for x in rounds if not x['void']][:a.max_pairs]
    out = dict(valid=len([x for x in rounds if not x['void']]), void=len([x for x in rounds if x['void']]),
               incomplete=len(incomplete), incomplete_rounds=incomplete,
               used=[x['round'] for x in ok], rounds=rounds, pairs={},
               runs={n: one(p, a) for n, p in (x.split('=', 1) for x in a.run)})
    for A, B in compares:
        P = [x for x in ok if A in x['runs'] and B in x['runs']]
        if not P:
            continue
        sv = np.array([100 * (1 - x['runs'][A]['secs'] / x['runs'][B]['secs']) for x in P])
        med = lambda arm, k: float(np.median([x['runs'][arm][k] for x in P]))
        o = dict(n=len(P), savings=sv.tolist(), median_saving=float(np.median(sv)), mean_saving=float(sv.mean()),
                 std_saving=float(sv.std(ddof=1)) if len(sv) > 1 else 0.0, faster=int((sv > 0).sum()),
                 sign_p=sign_test(sv),
                 time_ratio=float(np.median([x['runs'][A]['secs'] / x['runs'][B]['secs'] for x in P])),
                 reads_saved=100 * (1 - med(A, 'reads') / med(B, 'reads')))
        for k in ('s_per_step', 'reads_per_call', 'p50_ms', 'p99_ms', 'peak_mb', 'qps'):
            o[k] = {A: med(A, k), B: med(B, k)}
        o['p50_fall'] = 100 * (1 - o['p50_ms'][A] / o['p50_ms'][B])
        o['p99_fall'] = 100 * (1 - o['p99_ms'][A] / o['p99_ms'][B])
        out['pairs'][f'{A}:{B}'] = o
        print(f"{A} vs {B}: pairs {o['n']}  s/step {o['s_per_step'][B]:.2f} / {o['s_per_step'][A]:.2f}  "
              f"reads saved {o['reads_saved']:.2f}%  time saved {o['median_saving']:.2f}%  "
              f"faster {o['faster']}/{o['n']}  "
              f"sign p {o['sign_p']:.4g}  p50 fall {o['p50_fall']:.1f}%  p99 fall {o['p99_fall']:.1f}%")
    print(f"valid {out['valid']}  void {out['void']}  incomplete {out['incomplete']}  used {out['used']}")
    for x in incomplete:
        print(f"incomplete round {x['round']}: no summary line in the run(s) of {', '.join(x['missing'])}")
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
