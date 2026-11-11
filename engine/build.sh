#!/usr/bin/env bash
# Build the engine binaries from a DiskANN checkout at the pinned commit: the logging application (stock 4 KB pages),
# the read-once scoped engine and the hybrid-cache engine (8 KB pages), plus optional variants on separate copies.
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: build.sh --diskann <checkout> --work <dir> [options] [-- <extra cmake arguments>]
  --diskann DIR     DiskANN checkout at the pinned commit (fetch_sources.sh)
  --work DIR        directory for the patched trees and their builds
  --page-size N     page size of the scoped and hybrid engines in bytes (default 8192)
  --cxx-flags STR   extra compiler flags, e.g. include paths of Boost, MKL and libaio (default empty)
  --jobs N          parallel make jobs (default 8)
  --order-log       also build hybrid_search with the arrival-order log (copy hybrid_order)
  --aio-events N    also build hybrid_search with N asynchronous I/O events per thread (copy hybrid_aio)
Arguments after -- go to cmake, e.g. -DMKL_PATH=... -DBOOST_ROOT=... -DOMP_PATH=... -DCMAKE_CXX_STANDARD_LIBRARIES=-latomic
EOF
    exit 1
}

DISKANN=; WORK=; PAGE=8192; CXXF=; JOBS=8; ORDER=0; AIO=
while [ $# -gt 0 ]; do
    case $1 in
        --diskann) DISKANN=$2; shift 2;;
        --work) WORK=$2; shift 2;;
        --page-size) PAGE=$2; shift 2;;
        --cxx-flags) CXXF=$2; shift 2;;
        --jobs) JOBS=$2; shift 2;;
        --order-log) ORDER=1; shift;;
        --aio-events) AIO=$2; shift 2;;
        --) shift; break;;
        *) usage;;
    esac
done
[ -n "$DISKANN" ] && [ -n "$WORK" ] || usage
CMAKE_EXTRA=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
apply_edit() { python3 "$HERE/patches/$1.py" "${@:2}"; }   # edit script, tree, [options]
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)

copy_tree() {   # source, destination: the source tree without git metadata and build directories
    rm -rf "$2"; mkdir -p "$2"
    rsync -a --exclude /.git --exclude '/build*' "$1/" "$2/"
}

build() {   # tree, build dir name, page size (empty = stock), targets...
    local tree=$1 bdir=$2 unit=$3; shift 3
    local flags="$CXXF"; [ -n "$unit" ] && flags="$flags -DDISKANN_SECTOR_LEN=$unit"
    mkdir -p "$tree/$bdir"
    (cd "$tree/$bdir" && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$flags" ${CMAKE_EXTRA[@]+"${CMAKE_EXTRA[@]}"} \
        && make -j"$JOBS" "$@")
    for t in "$@"; do ls -l "$tree/$bdir/apps/$t"; done
}

# 1. logging application on the stock tree: stock 4 KB pages, stock layout (writes every traversal log)
copy_tree "$DISKANN" "$WORK/logger"
apply_edit add_logger "$WORK/logger"
build "$WORK/logger" build "" search_and_log

# 2. read-once scoped engine: layout map, page size at compile time, scoped page buffer
copy_tree "$WORK/logger" "$WORK/engine"
apply_edit add_layout_map "$WORK/engine"
apply_edit add_scoped_buffer "$WORK/engine"
build "$WORK/engine" build_$PAGE "$PAGE" batch_search search_and_log

# 3. hybrid-cache engine (every timed run)
copy_tree "$WORK/engine" "$WORK/hybrid"
apply_edit add_hybrid_cache "$WORK/hybrid"
build "$WORK/hybrid" build_$PAGE "$PAGE" hybrid_search

# 4. optional variants, each on its own copy so that the timed binary stays unchanged
if [ "$ORDER" = 1 ]; then
    copy_tree "$WORK/hybrid" "$WORK/hybrid_order"
    apply_edit add_order_log "$WORK/hybrid_order"
    build "$WORK/hybrid_order" build_$PAGE "$PAGE" hybrid_search
fi
if [ -n "$AIO" ]; then
    copy_tree "$WORK/hybrid" "$WORK/hybrid_aio"
    apply_edit set_aio_events "$WORK/hybrid_aio" --events "$AIO"
    build "$WORK/hybrid_aio" build_$PAGE "$PAGE" hybrid_search
fi
echo "engine binaries under $WORK/*/build*/apps"
