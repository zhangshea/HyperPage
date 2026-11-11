#!/usr/bin/env bash
# [Paper] Section VII-C, Other workloads and page size: HyperPage against MARGO on W1 and W2 under the hybrid cache
# (MARGO's share chosen on each draw), the cache-blind layouts on DEEP1M, and W3-1M with B = 6, 3 and 2 records per
# page.
set -euo pipefail

DATA_DIR=data            # DATA_DIR/<w>/splits.npz (W2 also calls.jsonl.gz; W3 also calls.jsonl.gz, drop.json)
HYPER_DIR=hyper          # HYPER_DIR/<w>/{val,test}_Q.npz and train{0,1,2}_U.npz
LAYOUT_DIR=layouts       # LAYOUT_DIR/<w>/<label>... (file names in experiments/README.md)
OUT_DIR=results/other_workloads
PY=python
WORKERS=8

DRAWS=0,1,2
SHARES=0,0.1,0.25,0.5,0.75,0.9,1
# the candidate order breaks exact ties between two candidates (after MARGO and the smaller share)

cd "$(dirname "$0")/.."
mkdir -p "$OUT_DIR"

schedules() {   # schedules W WORKLOAD [extra make_schedule arguments]
  local w=$1 workload=$2; shift 2
  for split in val test; do
    $PY replay/make_schedule.py "$workload" --split $split --sets "$HYPER_DIR/$w/${split}_Q.npz" \
      --splits "$DATA_DIR/$w/splits.npz" "$@" --out "$OUT_DIR/schedule_${w}_$split.npz"
  done
}

column() {   # column TAG W PAGES CAPACITY BUDGET CANDIDATES LAYOUT...: validation, selection, test at the chosen shares
  local tag=$1 w=$2 pages=$3 cap=$4 b=$5 cand=$6; shift 6
  local o=$OUT_DIR/${tag}_$b layouts=()
  for x in "$@"; do layouts+=(--layout "$x"); done
  common=(--train-unions "$HYPER_DIR/$w/train{draw}_U.npz" "${layouts[@]}" --draws $DRAWS --pages "$pages"
          --capacity "$cap" --budgets "$b" --workers $WORKERS)
  $PY replay/replay_hybrid.py --sets "$HYPER_DIR/$w/val_Q.npz" --schedule "$OUT_DIR/schedule_${w}_val.npz" \
    "${common[@]}" --shares $SHARES --out "$o.val.json"
  $PY replay/select_deployment.py "$o.val.json" --candidates "$cand" --out "$o.selection.json"
  $PY replay/replay_hybrid.py --sets "$HYPER_DIR/$w/test_Q.npz" --schedule "$OUT_DIR/schedule_${w}_test.npz" \
    "${common[@]}" --at "$o.selection.json" --out "$o.test.json"
}

margo_per_draw() {   # margo_per_draw TAG W BUDGET: MARGO's test reads at every share, then savings against MARGO
  local tag=$1 w=$2 b=$3 o=$OUT_DIR/$1_$3        # at the share chosen on each draw
  $PY replay/replay_hybrid.py --sets "$HYPER_DIR/$w/test_Q.npz" --schedule "$OUT_DIR/schedule_${w}_test.npz" \
    --train-unions "$HYPER_DIR/$w/train{draw}_U.npz" --layout "MARGO=$LAYOUT_DIR/$w/MARGO_B12.npy" --draws $DRAWS \
    --pages 83334 --budgets "$b" --shares $SHARES --workers $WORKERS --out "$o.test_margo.json"
  $PY replay/saving_table.py --selection "$o.selection.json" --test "$o.test.json" "$o.test_margo.json" \
    --incumbent-per-draw "$o.val.json" --out "$o.saving.json"
}

blind() {   # blind DIR MARGO_FILE: MARGO and the four cache-blind layouts of a directory
  echo "MARGO=$1/$2" "SET-G=$1/SET-G_r{draw}.npy" "ARC-G=$1/ARC-G_r{draw}.npy" "CLQ-G=$1/CLQ-G_r{draw}.npy" \
    "P05-G=$1/P05-G_r{draw}.npy"
}

# W1 and W2 on their own 1M indexes (83,334 pages), trained on their own runs. Candidates: MARGO, the four cache-blind
# layouts and P05-C with h = floor(rho_MARGO * D) hot pages; on W1 at 1% MARGO's share is 0, so h = 0 and P05-C is
# P05-G. W1 steps are 226 trajectories (make_schedule.py default).
schedules w1 w1
L=$LAYOUT_DIR/w1
column w1 w1 83334 12 833 ARC-G,CLQ-G,P05-G,SET-G $(blind "$L" MARGO_B12.npy)
margo_per_draw w1 w1 833
# ARC-G at 1% on W1 against MARGO at its share averaged over the draws
$PY replay/saving_table.py --selection "$OUT_DIR/w1_833.selection.json" --test "$OUT_DIR/w1_833.test.json" \
  --out "$OUT_DIR/w1_833.saving_shared_share.json"
for bh in "4167 1041" "20834 18750"; do
  set -- $bh
  column w1 w1 83334 12 "$1" ARC-G,CLQ-G,P05-C,P05-G,SET-G $(blind "$L" MARGO_B12.npy) "P05-C=$L/P05-C_h$2_r{draw}.npy"
  margo_per_draw w1 w1 "$1"
done
schedules w2 w2 --calls "$DATA_DIR/w2/calls.jsonl.gz"
L=$LAYOUT_DIR/w2
for bh in "833 416" "4167 3125" "20834 15625"; do
  set -- $bh
  column w2 w2 83334 12 "$1" ARC-G,CLQ-G,P05-C,P05-G,SET-G $(blind "$L" MARGO_B12.npy) "P05-C=$L/P05-C_h$2_r{draw}.npy"
  margo_per_draw w2 w2 "$1"
done

# DEEP1M (83,334 pages): cache-blind layouts only, each at its own validation share; 30 validation and 130 test steps
# of equal size in split order
schedules deep deep
L=$LAYOUT_DIR/deep
for b in 833 4167 20834; do
  column deep deep 83334 12 $b ARC-G,P05-G "MARGO=$L/MARGO_B12.npy" "ARC-G=$L/ARC-G_r{draw}.npy" \
    "P05-G=$L/P05-G_r{draw}.npy"
  $PY replay/saving_table.py --selection "$OUT_DIR/deep_$b.selection.json" --test "$OUT_DIR/deep_$b.test.json" \
    --out "$OUT_DIR/deep_$b.saving.json"
done

# W3-1M at B records per page: ceil(1,000,000 / B) pages, budgets round(f * pages) for f = 1%, 5%, 25%, layouts in
# LAYOUT_DIR/w3_b<B>. The cache-aware candidate refines the cache-blind objective with the fewest validation reads:
# P05 at B = 6 and 3, CLQ at B = 2 (layouts/README.md).
schedules w3 w3 --calls "$DATA_DIR/w3/calls.jsonl.gz" --drop "$DATA_DIR/w3/drop.json"
page_size() {   # page_size B PAGES CA_LABEL CANDIDATES "BUDGET H"...
  local B=$1 pages=$2 ca=$3 cand=$4 L=$LAYOUT_DIR/w3_b$1; shift 4
  for bh in "$@"; do
    set -- $bh
    column w3_b$B w3 "$pages" "$B" "$1" "$cand" $(blind "$L" MARGO_B$B.npy) "$ca=$L/${ca}_h$2_r{draw}.npy"
    $PY replay/saving_table.py --selection "$OUT_DIR/w3_b${B}_$1.selection.json" \
      --test "$OUT_DIR/w3_b${B}_$1.test.json" --out "$OUT_DIR/w3_b${B}_$1.saving.json"
  done
}
page_size 6 166667 P05-C ARC-G,P05-C,CLQ-G,P05-G,SET-G "1667 833" "8333 6249" "41667 10416"
page_size 3 333334 P05-C ARC-G,P05-C,CLQ-G,P05-G,SET-G "3333 2499" "16667 8333" "83334 41667"
page_size 2 500000 CLQ-C ARC-G,CLQ-C,CLQ-G,P05-G,SET-G "5000 3750" "25000 12500" "125000 93750"
