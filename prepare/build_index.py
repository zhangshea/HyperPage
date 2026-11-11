"""Vamana disk index built with MARGO's builder (R = 64, build list 125), which also writes MARGO's edge weights."""
import argparse
import os
import subprocess
import time

from hyperpage import formats

# [Paper] Section VII-A, Implementation details: the fixed Vamana graph, built with MARGO's builder, R = 64, build list
# size 125, 32-byte PQ codes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--builder', required=True, help="path of MARGO's tests/my_build_disk_index")
    ap.add_argument('--data-type', choices=['int8', 'float'], default='int8')
    ap.add_argument('--data', required=True, help='base.bin (int8) or base.fbin (float)')
    ap.add_argument('--index-prefix', required=True, help='output prefix, e.g. INDEX_DIR/margo_R64_L125')
    ap.add_argument('--pq-budget-gb', default='0.03',
                    help='-B: PQ budget in GB, chosen so that floor(B * 2^30 / n) = 32 bytes per record '
                         '(0.03 at 1M, 0.15 at 5M, 0.30 at 10M, 0.64 at 21M)')
    ap.add_argument('--build-mem-gb', default='32', help='-M: build memory in GB (32 at 1M, 64 at 5M and above)')
    ap.add_argument('--threads', type=int, default=12, help='-T')
    ap.add_argument('--log', default=None, help='file for the builder output (default: <index-prefix>_build.log)')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.index_prefix)), exist_ok=True)
    cmd = [a.builder, '--data_type', a.data_type, '--dist_fn', 'l2', '--data_path', a.data,
           '--index_path_prefix', a.index_prefix, '-R', '64', '-L', '125', '-B', a.pq_budget_gb, '-M', a.build_mem_gb,
           '-T', str(a.threads)]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env=dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads)))
    open(a.log or a.index_prefix + '_build.log', 'w').write(r.stdout + '\n' + r.stderr)
    print('build exit', r.returncode, 'secs', round(time.time() - t, 1))
    if r.returncode != 0:
        raise SystemExit(r.stderr[-2000:])
    print(formats.read_index_meta(a.index_prefix + '_disk.index'))


if __name__ == '__main__':
    main()
