"""Read speed of index files: the same uniformly random page offsets read from each file with O_DIRECT, one read at a
time per thread, each thread pinned to one CPU; files are measured in a rotated order over several repetitions."""
# [Paper] Section VII-B, Search time: a pair is void when a check fixed before the run flags the read speed of the two
# files (read_check.py applies the rule to this output). Reported per file: mean and median latency, reads per second.
import argparse
import json
import mmap
import os
import threading
import time

import numpy as np


def cpu_list(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-'); out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('files', nargs='+', help='index files (<prefix>_disk.index)')
    ap.add_argument('--cpus', required=True, help='CPUs of the timed runs, one reading thread per CPU, e.g. 0-11')
    ap.add_argument('--reads', type=int, default=20000, help='reads per file and repetition')
    ap.add_argument('--repeats', type=int, default=3, help='repetitions')
    ap.add_argument('--page-size', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=7, help='seed of the page offsets')
    a = ap.parse_args()
    PAGE = a.page_size; CPUS = cpu_list(a.cpus); NT = len(CPUS); FILES = a.files
    npg = min(os.path.getsize(f) // PAGE for f in FILES) - 1
    offs = (1 + np.random.default_rng(a.seed).integers(0, npg, a.reads)) * PAGE   # page 0 is the header
    res = {f: [] for f in FILES}

    def worker(fd, part, lat, cpu, errors):
        try:
            os.sched_setaffinity(0, {cpu})
            buf = mmap.mmap(-1, PAGE)   # page-aligned buffer, as O_DIRECT requires
            for i, o in enumerate(part):
                t = time.perf_counter_ns(); n = os.preadv(fd, [buf], int(o)); lat[i] = time.perf_counter_ns() - t
                if n != PAGE:
                    raise OSError(f'short read: {n} of {PAGE} bytes at offset {int(o)}')
        except BaseException as e:   # handed to the main thread, which stops the measurement
            errors.append(e)

    for r in range(a.repeats):
        order = FILES[r % len(FILES):] + FILES[:r % len(FILES)]
        for f in order:
            fd = os.open(f, os.O_RDONLY | os.O_DIRECT)
            parts = np.array_split(offs, NT); lats = [np.zeros(len(p), np.int64) for p in parts]
            errors = []
            th = [threading.Thread(target=worker, args=(fd, parts[k], lats[k], CPUS[k], errors)) for k in range(NT)]
            t0 = time.perf_counter()
            for t in th:
                t.start()
            for t in th:
                t.join()
            secs = time.perf_counter() - t0; os.close(fd)
            if errors:
                raise RuntimeError(f'{f}: a reading thread failed') from errors[0]
            lat = np.concatenate(lats) / 1e3
            res[f].append(dict(rep=r, mean_us=float(lat.mean()), median_us=float(np.median(lat)),
                               p99_us=float(np.percentile(lat, 99)), reads_per_s=a.reads / secs))
    out = {f: dict(mean_us=float(np.median([x['mean_us'] for x in v])),
                   median_us=float(np.median([x['median_us'] for x in v])),
                   reads_per_s=float(np.median([x['reads_per_s'] for x in v])), reps=v) for f, v in res.items()}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
