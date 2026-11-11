"""CPU-affinity check of a timed run: snapshots of the user processes before and after the run, and a flag when a
foreign process that may run on the run's CPUs used more than a share of the run's wall time (reads /proc only)."""
# snap OUT              writes {pid: [comm, uid, cpu_seconds, cpus_allowed]} for every user process; kernel threads
#                       (empty cmdline) are skipped, since per-CPU kernel threads serve the run's own I/O.
# diff A B WALL [PID..] flags the run when a process (not excluded, not this script) whose CPU affinity meets the run's
#                       CPUs used more than --max-share of WALL seconds between snapshots A and B (all of its CPU time
#                       if it started after A). Prints JSON {flag, wall, offenders, excluded}.
# [Paper] Section VII-B, Search time: a pair is void when a check fixed before the run flags CPU affinity.
import argparse
import json
import os


def cpus(s):
    out = set()
    for part in s.strip().split(','):
        if part:
            a, _, b = part.partition('-'); out.update(range(int(a), int(b or a) + 1))
    return out


def snap():
    hz = os.sysconf('SC_CLK_TCK'); d = {}
    for p in os.listdir('/proc'):
        if not p.isdigit():
            continue
        try:
            if not open(f'/proc/{p}/cmdline', 'rb').read():
                continue
            st = open(f'/proc/{p}/stat').read()
            comm = st[st.index('(') + 1:st.rindex(')')]; f = st[st.rindex(')') + 2:].split()
            cpu_s = (int(f[11]) + int(f[12])) / hz
            stat = dict(line.split(':', 1) for line in open(f'/proc/{p}/status') if ':' in line)
            d[p] = [comm, stat.get('Uid', '').split()[0] if stat.get('Uid') else '', cpu_s,
                    stat.get('Cpus_allowed_list', '').strip()]
        except (OSError, ValueError, IndexError):
            continue
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('snap', help='write a snapshot of the user processes')
    s.add_argument('out')
    d = sub.add_parser('diff', help='compare two snapshots taken around a run')
    d.add_argument('before'); d.add_argument('after')
    d.add_argument('wall', type=float, help='wall seconds of the run')
    d.add_argument('exclude', nargs='*', help='pids to leave out (the run and its driver)')
    d.add_argument('--cpus', required=True, help='CPUs of the run, e.g. 0-11')
    d.add_argument('--max-share', type=float, default=0.01, help='largest CPU time of a foreign process, share of wall')
    d.add_argument('--exempt', default='',
                   help='comma-separated process names kept on record but never flagged, e.g. a resident metrics '
                        'exporter of the host (default: none)')
    a = ap.parse_args()
    if a.cmd == 'snap':
        json.dump(snap(), open(a.out, 'w'))
        return
    timed = cpus(a.cpus); exempt = set(filter(None, a.exempt.split(',')))
    A, B = json.load(open(a.before)), json.load(open(a.after)); excl = set(a.exclude)
    me = str(os.getpid()); off = []; exc = []
    for p, (comm, uid, c1, aff) in B.items():
        if p in excl or p == me or not (cpus(aff) & timed):
            continue
        c0 = A[p][2] if p in A and A[p][0] == comm else 0.0
        if comm in exempt:
            exc.append(dict(pid=p, comm=comm, cpu_s=round(c1 - c0, 2)))
        elif c1 - c0 > a.max_share * a.wall:
            off.append(dict(pid=p, comm=comm, uid=uid, cpu_s=round(c1 - c0, 2), cpus=aff))
    print(json.dumps(dict(flag=bool(off), wall=a.wall, offenders=off, excluded=exc)))


if __name__ == '__main__':
    main()
