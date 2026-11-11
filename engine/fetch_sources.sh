#!/usr/bin/env bash
# Fetch the upstream sources the engine builds on, at the pinned commits: DiskANN (MIT) and, for Table V only, MARGO.
# Usage: fetch_sources.sh <dest dir> [--with-margo]
set -euo pipefail

DISKANN_URL=https://github.com/microsoft/DiskANN.git
DISKANN_BRANCH=cpp_main
DISKANN_COMMIT=78256bbab4685e1774e78d331e081a153be26823
MARGO_URL=https://github.com/CodenameYZY/MARGO.git
MARGO_COMMIT=7e24023b950c1a31d993476a40f444dd69617ba7

[ $# -ge 1 ] || { echo "usage: $0 <dest dir> [--with-margo]" >&2; exit 1; }
DEST=$1; WITH_MARGO=${2:-}
mkdir -p "$DEST"

fetch() {   # url, directory, commit, [branch]
    local url=$1 dir=$2 commit=$3 branch=${4:-}
    if [ ! -d "$dir/.git" ]; then
        if [ -n "$branch" ]; then git clone --branch "$branch" --single-branch "$url" "$dir"; else git clone "$url" "$dir"; fi
    fi
    git -C "$dir" checkout --detach "$commit"
    echo "$dir at $(git -C "$dir" rev-parse HEAD)"
}

fetch "$DISKANN_URL" "$DEST/diskann" "$DISKANN_COMMIT" "$DISKANN_BRANCH"
if [ "$WITH_MARGO" = --with-margo ]; then
    # MARGO's repository states no licence: it is fetched from its authors and edited locally, never redistributed
    fetch "$MARGO_URL" "$DEST/margo" "$MARGO_COMMIT"
fi
