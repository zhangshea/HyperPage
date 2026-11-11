#!/usr/bin/env bash
# Build MARGO's page-search engine with the 8 KB port (patches/port_margo_page_search.py) from a local MARGO checkout.
# Usage: build_margo.sh --margo <checkout> --work <dir> [--cxx-flags STR] [--jobs N] [--aio-events N] [-- cmake args]
set -euo pipefail

MARGO=; WORK=; CXXF=; JOBS=8; AIO=
while [ $# -gt 0 ]; do
    case $1 in
        --margo) MARGO=$2; shift 2;;
        --work) WORK=$2; shift 2;;
        --cxx-flags) CXXF=$2; shift 2;;
        --jobs) JOBS=$2; shift 2;;
        --aio-events) AIO=$2; shift 2;;
        --) shift; break;;
        *) echo "unknown argument $1" >&2; exit 1;;
    esac
done
[ -n "$MARGO" ] && [ -n "$WORK" ] || { echo "usage: $0 --margo <checkout> --work <dir> [...] [-- cmake args]" >&2; exit 1; }
CMAKE_EXTRA=("$@")
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)

one() {   # tree name, AIO events (empty = unchanged)
    local T=$WORK/$1
    rm -rf "$T"; mkdir -p "$T"; rsync -a --exclude /.git --exclude /release "$MARGO/" "$T/"
    python3 "$HERE/patches/port_margo_page_search.py" "$T"
    [ -n "$2" ] && python3 "$HERE/patches/set_aio_events.py" "$T" --events "$2"
    mkdir -p "$T/release"
    (cd "$T/release" && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$CXXF" ${CMAKE_EXTRA[@]+"${CMAKE_EXTRA[@]}"} \
        && make -j"$JOBS" search_disk_index index_relayout)
    ls -l "$T/release/tests/search_disk_index" "$T/release/tests/utils/index_relayout"
}

one margo_page_search ""
[ -n "$AIO" ] && one margo_page_search_aio "$AIO"
echo done
