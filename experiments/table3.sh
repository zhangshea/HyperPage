#!/usr/bin/env bash
# [Paper] Table III: saving over MARGO under equal cache budgets (W3-1M, W3-21M, W4 with the W3-1M layouts), MARGO's
# reads per step, HyperPage's deployment, the bootstrap of every margin (Section VII-A) and the last row, the budget
# at which MARGO reads as few pages as HyperPage. Table VII and Fig. 4 reuse the files written here.
set -euo pipefail

DATA_DIR=data            # DATA_DIR/<w>/calls.jsonl.gz, splits.npz (W3: drop.json; W4: qstr.json, meta_<split>.npy)
HYPER_DIR=hyper          # HYPER_DIR/<w>/{val,test}_Q.npz and train{0,1,2}_U.npz
LAYOUT_DIR=layouts       # LAYOUT_DIR/<w>/<label>... (file names in experiments/README.md)
OUT_DIR=results/table3
PY=python
WORKERS=8                # processes per replay call

DRAWS=0,1,2
SHARES=0,0.1,0.25,0.5,0.75,0.9,1     # hot-record share grid rho
# HyperPage's candidates besides MARGO, in every column; the order breaks exact ties between two candidates
CANDIDATES=ARC-C,ARC-CF,ARC-G,CLQ-G,HEAT-res,MARGO-uncached,P05-C,P05-G,SET-G,STARLING
# validation: 10 warm-up steps, then 20 scored steps (replay_hybrid.py and select_deployment.py defaults). W3 has 30
# validation steps (41 to 70); W4 has 88 (8 checkpoints x 11 blocks of 1,024 questions), of which steps 11 to 30 are
# scored.

cd "$(dirname "$0")/.."
mkdir -p "$OUT_DIR"

# layout file of a label: one file per draw ({draw}), cache-aware layouts also per hot-page count h
lay() {   # lay DIR LABEL [h]
  case $2 in
    MARGO) echo "MARGO=$1/MARGO_B12.npy" ;;
    STARLING|RCM|Proximity) echo "$2=$1/$2.npy" ;;
    *) if [ $# -eq 3 ]; then echo "$2=$1/$2_h$3_r{draw}.npy"; else echo "$2=$1/$2_r{draw}.npy"; fi ;;
  esac
}

schedules() {   # schedules W WORKLOAD: arrival metadata of the validation and test splits of directory W
  local w=$1 workload=$2
  for split in val test; do
    extra=""
    [ "$workload" = w4 ] && extra="--strings $DATA_DIR/w4/qstr.json --questions $DATA_DIR/w4/meta_$split.npy"
    [ "$workload" = w3 ] && extra="--drop $DATA_DIR/$w/drop.json"   # groups that make_splits.py removed
    $PY replay/make_schedule.py "$workload" --split $split --sets "$HYPER_DIR/$w/${split}_Q.npz" \
      --splits "$DATA_DIR/$w/splits.npz" --calls "$DATA_DIR/$w/calls.jsonl.gz" $extra \
      --out "$OUT_DIR/schedule_${w}_$split.npz"
  done
}

column() {   # column TAG EVAL_W TRAIN_W PAGES BUDGET H: one column of Tables III and VII
  local tag=$1 w=$2 tw=$3 pages=$4 b=$5 h=$6 L=$LAYOUT_DIR/$3 o=$OUT_DIR/$1_$5
  layouts=()
  for x in MARGO STARLING RCM Proximity FREQ SHP-G Mt-KaHyPar HEAT SET-G ARC-G CLQ-G P05-G; do
    # the full-corpus RCM and Proximity layouts have no build command in layouts/, so their columns skip them
    if [ "$tw" = w21 ] && { [ $x = RCM ] || [ $x = Proximity ]; }; then continue; fi
    layouts+=(--layout "$(lay "$L" $x)"); done
  for x in Mt-KaHyPar-res HEAT-res ARC-C ARC-CF P05-C MARGO-uncached; do
    layouts+=(--layout "$(lay "$L" $x "$h")"); done
  common=(--train-unions "$HYPER_DIR/$tw/train{draw}_U.npz" "${layouts[@]}" --draws $DRAWS --pages "$pages"
          --budgets "$b" --workers $WORKERS)
  # validation: every layout at every share of the grid
  $PY replay/replay_hybrid.py --sets "$HYPER_DIR/$w/val_Q.npz" --schedule "$OUT_DIR/schedule_${w}_val.npz" \
    "${common[@]}" --shares $SHARES --out "$o.val.json"
  # each layout's share and HyperPage's (layout, share) of every draw
  $PY replay/select_deployment.py "$o.val.json" --candidates $CANDIDATES --out "$o.selection.json"
  # test: every layout at its validation share and at the share of every draw that deploys it
  $PY replay/replay_hybrid.py --sets "$HYPER_DIR/$w/test_Q.npz" --schedule "$OUT_DIR/schedule_${w}_test.npz" \
    "${common[@]}" --at "$o.selection.json" --out "$o.test.json"
  $PY replay/saving_table.py --selection "$o.selection.json" --test "$o.test.json" --out "$o.saving.json"
  $PY replay/significance.py --selection "$o.selection.json" --test "$o.test.json" --out "$o.significance.json"
}

margo_grid() {   # margo_grid TAG EVAL_W TRAIN_W PAGES GRID INDEX_PAGES: MARGO over the budget grid, every share
  local o=$OUT_DIR/$1_grid
  budgets=$($PY replay/budget_match.py --grid "$5" --index-pages "$6" --list-budgets)
  for split in val test; do
    $PY replay/replay_hybrid.py --sets "$HYPER_DIR/$2/${split}_Q.npz" --schedule "$OUT_DIR/schedule_$2_$split.npz" \
      --train-unions "$HYPER_DIR/$3/train{draw}_U.npz" --layout "$(lay "$LAYOUT_DIR/$3" MARGO)" --draws $DRAWS \
      --pages "$4" --budgets "$budgets" --shares $SHARES --workers $WORKERS --out "$o.$split.json"
  done
}

match() {   # match TAG BUDGET DECLARED_FRACTION GRID INDEX_PAGES: last row of Table III
  $PY replay/budget_match.py --grid "$4" --index-pages "$5" --declared "$3" \
    --validation "$OUT_DIR/$1_grid.val.json" --test "$OUT_DIR/$1_grid.test.json" \
    --selection "$OUT_DIR/$1_$2.selection.json" --deployed-test "$OUT_DIR/$1_$2.test.json" \
    --out "$OUT_DIR/$1_$2.budget_match.json"
}

# W3-1M: 83,334 pages; budgets 1%, 5%, 25% = 833, 4167, 20834 pages; h = floor(rho_MARGO * D) = 416, 3750, 15625
schedules w3 w3
column w3-1m w3 w3 83334 833 416
column w3-1m w3 w3 83334 4167 3750
column w3-1m w3 w3 83334 20834 15625
margo_grid w3-1m w3 w3 83334 1m 83334
match w3-1m 833 0.01 1m 83334
match w3-1m 4167 0.05 1m 83334
match w3-1m 20834 0.25 1m 83334

# W3-21M: the index has 1,751,277 pages; budgets round(f * 1,751,278) for 1%, 5%, 25% and 2^31 / 8192 for 15%:
# 17513, 87564, 262144, 437820 pages; h = 8756, 21891, 196608, 394038. The replay's page arrays hold 1,751,278 pages
# (one never used). The W3 sub-queries are the same as at 1M.
schedules w21 w3
column w3-21m w21 w21 1751278 17513 8756
column w3-21m w21 w21 1751278 87564 21891
column w3-21m w21 w21 1751278 262144 196608
column w3-21m w21 w21 1751278 437820 394038
margo_grid w3-21m w21 w21 1751278 full-corpus 1751278
match w3-21m 17513 0.01 full-corpus 1751278
match w3-21m 87564 0.05 full-corpus 1751278
match w3-21m 262144 0.149687 full-corpus 1751278
match w3-21m 437820 0.25 full-corpus 1751278

# W4: OLMo-3 proxy groups replayed on the 1M index with the W3-1M layouts and hot-record rankings
schedules w4 w4
column w4 w4 w3 83334 833 416
column w4 w4 w3 83334 4167 3750
column w4 w4 w3 83334 20834 15625
margo_grid w4 w4 w3 83334 1m 83334
match w4 833 0.01 1m 83334
match w4 4167 0.05 1m 83334
match w4 20834 0.25 1m 83334
