#!/usr/bin/env bash
# [Paper] Fig. 3: saving of the four cache-blind objectives over MARGO on W3 test when every layout uses the same
# hot-record share, with no selection. The test steps are replayed at every share of the grid for all seven budgets;
# the figure shows 1M at 1% and the full corpus at 5% and 25%, and the share-0 rows give the pure page LRU numbers of
# Section VII-B.
set -euo pipefail

DATA_DIR=data            # DATA_DIR/<w>/calls.jsonl.gz, splits.npz, drop.json
HYPER_DIR=hyper
LAYOUT_DIR=layouts
OUT_DIR=results/fig3
PY=python
WORKERS=8

cd "$(dirname "$0")/.."
mkdir -p "$OUT_DIR"
for scale in "w3 83334 833,4167,20834" "w21 1751278 17513,87564,262144,437820"; do
  set -- $scale; w=$1; pages=$2; budgets=$3; L=$LAYOUT_DIR/$w
  $PY replay/make_schedule.py w3 --split test --sets "$HYPER_DIR/$w/test_Q.npz" --splits "$DATA_DIR/$w/splits.npz" \
    --calls "$DATA_DIR/$w/calls.jsonl.gz" --drop "$DATA_DIR/$w/drop.json" --out "$OUT_DIR/schedule_${w}_test.npz"
  $PY replay/replay_hybrid.py --sets "$HYPER_DIR/$w/test_Q.npz" --schedule "$OUT_DIR/schedule_${w}_test.npz" \
    --train-unions "$HYPER_DIR/$w/train{draw}_U.npz" --pages "$pages" --budgets "$budgets" \
    --shares 0,0.1,0.25,0.5,0.75,0.9,1 --draws 0,1,2 --workers $WORKERS \
    --layout "MARGO=$L/MARGO_B12.npy" --layout "SET-G=$L/SET-G_r{draw}.npy" --layout "ARC-G=$L/ARC-G_r{draw}.npy" \
    --layout "CLQ-G=$L/CLQ-G_r{draw}.npy" --layout "P05-G=$L/P05-G_r{draw}.npy" --layout "FREQ=$L/FREQ_r{draw}.npy" \
    --out "$OUT_DIR/test_$w.json"
done
# MARGO's hot set is that of each draw, so MARGO also has one row per draw
$PY experiments/plot_share_curves.py --out "$OUT_DIR/fig3.pdf" \
  --panel "1M, 1%" 833 "$OUT_DIR/test_w3.json" \
  --panel "21M, 5%" 87564 "$OUT_DIR/test_w21.json" \
  --panel "21M, 25%" 437820 "$OUT_DIR/test_w21.json"
# all seven budgets, with FREQ (pure page LRU = share 0)
$PY experiments/plot_share_curves.py --layouts SET-G,ARC-G,CLQ-G,P05-G,FREQ \
  --panel "1M, 1%" 833 "$OUT_DIR/test_w3.json" --panel "1M, 5%" 4167 "$OUT_DIR/test_w3.json" \
  --panel "1M, 25%" 20834 "$OUT_DIR/test_w3.json" --panel "21M, 1%" 17513 "$OUT_DIR/test_w21.json" \
  --panel "21M, 5%" 87564 "$OUT_DIR/test_w21.json" --panel "21M, 15%" 262144 "$OUT_DIR/test_w21.json" \
  --panel "21M, 25%" 437820 "$OUT_DIR/test_w21.json"
