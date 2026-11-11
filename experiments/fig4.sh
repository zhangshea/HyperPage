#!/usr/bin/env bash
# [Paper] Fig. 4: saving over MARGO on W3 test at 1% and 5% of the 1M, 5M, 10M and 21M indexes, for HyperPage (picks
# among MARGO, ARC-G, P05-G, ARC-C and P05-C) and the cache-blind objectives ARC-G and P05-G, each at its own share.
set -euo pipefail

DATA_DIR=data            # DATA_DIR/<w>: w3 (1M), w5m, w10m, w21; the larger ones hold W3's query-side files
HYPER_DIR=hyper
LAYOUT_DIR=layouts
OUT_DIR=results/fig4
PY=python
WORKERS=8
CANDIDATES=ARC-G,P05-G,ARC-C,P05-C   # besides MARGO; this order breaks exact ties between two candidates

cd "$(dirname "$0")/.."
mkdir -p "$OUT_DIR"
point() {   # point W PAGES BUDGET H
  local w=$1 pages=$2 b=$3 h=$4 L=$LAYOUT_DIR/$1 o=$OUT_DIR/$1_$3
  common=(--train-unions "$HYPER_DIR/$w/train{draw}_U.npz" --pages "$pages" --budgets "$b" --draws 0,1,2
          --shares 0,0.1,0.25,0.5,0.75,0.9,1 --workers $WORKERS --layout "MARGO=$L/MARGO_B12.npy"
          --layout "ARC-G=$L/ARC-G_r{draw}.npy" --layout "P05-G=$L/P05-G_r{draw}.npy"
          --layout "ARC-C=$L/ARC-C_h${h}_r{draw}.npy" --layout "P05-C=$L/P05-C_h${h}_r{draw}.npy")
  for split in val test; do
    [ -f "$OUT_DIR/schedule_${w}_$split.npz" ] ||
      $PY replay/make_schedule.py w3 --split $split --sets "$HYPER_DIR/$w/${split}_Q.npz" \
        --splits "$DATA_DIR/$w/splits.npz" --calls "$DATA_DIR/$w/calls.jsonl.gz" --drop "$DATA_DIR/$w/drop.json" \
        --out "$OUT_DIR/schedule_${w}_$split.npz"
  done
  $PY replay/replay_hybrid.py --sets "$HYPER_DIR/$w/val_Q.npz" --schedule "$OUT_DIR/schedule_${w}_val.npz" \
    "${common[@]}" --out "$o.val.json"
  $PY replay/select_deployment.py "$o.val.json" --candidates $CANDIDATES --out "$o.selection.json"
  # test over the whole share grid: a deployed share may differ from the layout's share averaged over the draws
  $PY replay/replay_hybrid.py --sets "$HYPER_DIR/$w/test_Q.npz" --schedule "$OUT_DIR/schedule_${w}_test.npz" \
    "${common[@]}" --out "$o.test.json"
  $PY replay/saving_table.py --selection "$o.selection.json" --test "$o.test.json" --out "$o.saving.json"
}
# budgets round(f * pages) at f = 1% and 5%; h = floor(rho_MARGO * D) of the cache-aware layouts
point w3   83334   833   416
point w3   83334   4167  3750
point w5m  416667  4167  3125
point w5m  416667  20833 15624
point w10m 833334  8333  6249
point w10m 833334  41667 20833
point w21  1751278 17513 8756
point w21  1751278 87564 21891
$PY experiments/plot_scale.py --out "$OUT_DIR/fig4.pdf" \
  --index 1M 1 "$OUT_DIR/w3_833.saving.json" "$OUT_DIR/w3_4167.saving.json" \
  --index 5M 5 "$OUT_DIR/w5m_4167.saving.json" "$OUT_DIR/w5m_20833.saving.json" \
  --index 10M 10 "$OUT_DIR/w10m_8333.saving.json" "$OUT_DIR/w10m_41667.saving.json" \
  --index 21M 21 "$OUT_DIR/w21_17513.saving.json" "$OUT_DIR/w21_87564.saving.json"
