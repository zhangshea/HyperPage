#!/usr/bin/env bash
# Run one timed arm of a pair: the hybrid-cache engine (hybrid_search) or MARGO's page-search engine, pinned to the given
# CPUs and bracketed by CPU-affinity snapshots; writes <out>.log, <out>.host, <out>.affinity.json and the engine outputs.
# [Paper] Table IV and Table V: one arm of a timed pair or round (Section VII-B, Search time).
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: run_arm.sh --engine hybrid|page-search --bin <binary> --cpus <list, e.g. 0-11> --threads <n> --out <prefix>
                  --queries <query.bin> --L <list size> [engine options]
hybrid:      --index <layout prefix> --wave-ptr <wptr.u64> --wave-step <wstep.u32> --hot <hot.u32> --lru <pages>
             --warm <warm query.bin> [--W 4]
page-search: --index-dir <dir holding m_disk.index> --cache-list <records.u32> --cache-records <n> --disk <block device
             name as in /proc/diskstats> [--W 1]
common:      [--data-type int8] [--metric l2] [--K 10] [--max-share 0.01]
             [--exempt <comma-separated process names never flagged by the affinity check; default none>]
EOF
    exit 1
}

ENGINE=; BIN=; CPUS=; NT=; OUT=; Q=; L=; IDX=; WPTR=; WSTEP=; HOT=; LRU=; WARM=; W=; IDXDIR=; CLIST=; CREC=; DISK=
DT=int8; METRIC=l2; K=10; MAXSHARE=0.01; EXEMPT=
while [ $# -gt 0 ]; do
    case $1 in
        --engine) ENGINE=$2;; --bin) BIN=$2;; --cpus) CPUS=$2;; --threads) NT=$2;; --out) OUT=$2;;
        --queries) Q=$2;; --L) L=$2;; --W) W=$2;; --K) K=$2;; --data-type) DT=$2;; --metric) METRIC=$2;;
        --index) IDX=$2;; --wave-ptr) WPTR=$2;; --wave-step) WSTEP=$2;; --hot) HOT=$2;; --lru) LRU=$2;; --warm) WARM=$2;;
        --index-dir) IDXDIR=$2;; --cache-list) CLIST=$2;; --cache-records) CREC=$2;; --disk) DISK=$2;;
        --max-share) MAXSHARE=$2;; --exempt) EXEMPT=$2;;
        *) usage;;
    esac
    shift 2
done
[ -n "$ENGINE" ] && [ -n "$BIN" ] && [ -n "$CPUS" ] && [ -n "$NT" ] && [ -n "$OUT" ] && [ -n "$Q" ] && [ -n "$L" ] || usage
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$(dirname "$OUT")"

status=0
python3 "$HERE/affinity_check.py" snap "$OUT.aff_a.json"
t0=$(date +%s.%N)
case $ENGINE in
    hybrid)
        [ -n "$IDX" ] && [ -n "$WPTR" ] && [ -n "$WSTEP" ] && [ -n "$HOT" ] && [ -n "$LRU" ] && [ -n "$WARM" ] || usage
        # one OpenMP place per CPU of the list, threads bound to them
        PLACES=$(python3 -c "import sys
c = []
for p in sys.argv[1].split(','):
    a, _, b = p.partition('-'); c += range(int(a), int(b or a) + 1)
print(','.join('{%d}' % x for x in c))" "$CPUS")
        OMP_PLACES=$PLACES OMP_PROC_BIND=true "$BIN" "$DT" "$METRIC" "$IDX" "$Q" "$WPTR" "$WSTEP" "$L" "${W:-4}" "$K" "$NT" \
            "$HOT" "$LRU" "$OUT" 0 "$WARM" frontier > "$OUT.log" 2>&1 &
        pid=$!; wait $pid || status=$?
        ;;
    page-search)
        [ -n "$IDXDIR" ] && [ -n "$CLIST" ] && [ -n "$CREC" ] && [ -n "$DISK" ] || usage
        { echo @@stat0; cat /proc/stat; echo @@disk0; cat /proc/diskstats; } > "$OUT.host"
        # the timed device sampled every 5 s: the per-read I/O time of the first and last fifth of the run
        ( while true; do echo "@@t $(date +%s.%N) $(grep " $DISK " /proc/diskstats)"; sleep 5; done ) > "$OUT.dsamp" &
        sp=$!
        RECORD_CACHE_LIST=$CLIST taskset -c "$CPUS" "$BIN" --data_type "$DT" --dist_fn "$METRIC" \
            --index_path_prefix "$IDXDIR/m" --query_file "$Q" --gt_file null -K "$K" --result_path "$OUT" \
            --num_nodes_to_cache "$CREC" -T "$NT" -L "$L" -W "${W:-1}" --mem_L 0 --use_page_search 1 --use_ratio 1.0 \
            --disk_file_path "$IDXDIR/m_disk.index" > "$OUT.log" 2>&1 &
        pid=$!; wait $pid || status=$?
        kill $sp 2>/dev/null || true; wait $sp 2>/dev/null || true
        { echo @@stat1; cat /proc/stat; echo @@disk1; cat /proc/diskstats; } >> "$OUT.host"
        rm -f "${OUT}"_*_dists_float.bin
        ;;
    *) usage;;
esac
t1=$(date +%s.%N)
python3 "$HERE/affinity_check.py" snap "$OUT.aff_b.json"
python3 "$HERE/affinity_check.py" diff "$OUT.aff_a.json" "$OUT.aff_b.json" "$(python3 -c "print($t1 - $t0)")" "$pid" $$ \
    --cpus "$CPUS" --max-share "$MAXSHARE" --exempt "$EXEMPT" > "$OUT.affinity.json"
grep -h "^summary\|^HPSTAT" "$OUT.log" | tr ' ' '\n' | grep -A1 -E '^(secs|page_reads|sum_ios|us_per_read|lat_ms_p99|p99_us)$' \
    | paste -s - || true
exit $status
