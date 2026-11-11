#!/usr/bin/env bash
# [Paper] Table VI: mean reads per W3 test group under the keyed group buffer and the saving Delta over MARGO per group
# and per tool call, at 1M and on the full corpus. The same group means give the ablation steps of Section VII-C.
set -euo pipefail

DATA_DIR=data            # DATA_DIR/<w>/calls.jsonl.gz, splits.npz, drop.json
HYPER_DIR=hyper          # HYPER_DIR/<w>/test_Q.npz, test_U.npz
LAYOUT_DIR=layouts       # file names in experiments/README.md
OUT_DIR=results/table6
PY=python

cd "$(dirname "$0")/.."
mkdir -p "$OUT_DIR"
for scale in "w3 83334" "w21 1751278"; do
  set -- $scale; w=$1; pages=$2; L=$LAYOUT_DIR/$w
  $PY replay/make_schedule.py w3 --split test --sets "$HYPER_DIR/$w/test_Q.npz" --splits "$DATA_DIR/$w/splits.npz" \
    --calls "$DATA_DIR/$w/calls.jsonl.gz" --drop "$DATA_DIR/$w/drop.json" --out "$OUT_DIR/schedule_${w}_test.npz"
  # STARLING is one layout; the logged layouts have one file per training draw, and Delta uses their mean reads
  $PY replay/scope_reads.py --unions "$HYPER_DIR/$w/test_U.npz" --sets "$HYPER_DIR/$w/test_Q.npz" \
    --schedule "$OUT_DIR/schedule_${w}_test.npz" --pages "$pages" --draws 0,1,2 \
    --layout "MARGO=$L/MARGO_B12.npy" --layout "STARLING=$L/STARLING.npy" --layout "FREQ=$L/FREQ_r{draw}.npy" \
    --layout "Mt-KaHyPar=$L/Mt-KaHyPar_r{draw}.npy" --layout "SHP-G=$L/SHP-G_r{draw}.npy" \
    --layout "ARC-G=$L/ARC-G_r{draw}.npy" --layout "CLQ-G=$L/CLQ-G_r{draw}.npy" \
    --layout "SET-G=$L/SET-G_r{draw}.npy" --layout "P05-G=$L/P05-G_r{draw}.npy" \
    --out "$OUT_DIR/scope_$w.json"
done
