#!/usr/bin/env bash
# [Paper] Section VII-B, Search time: W3-1M validation replayed in the arrival order that the 12-thread engine logs with
# MARGO's layout, and the selection of the timed 1M layouts at 1% and 5% under that order (Table IV, 1M 1% and 5%).
set -euo pipefail

DATA_DIR=data            # DATA_DIR/w3/calls.jsonl.gz, splits.npz, drop.json
HYPER_DIR=hyper          # HYPER_DIR/w3/val_Q.npz, train{0,1,2}_U.npz
LAYOUT_DIR=layouts       # LAYOUT_DIR/w3/<label>... (file names in experiments/README.md)
ENGINE_DIR=engine_runs   # ENGINE_DIR/<D>/val_lock.npy (stage_hybrid.py) and ENGINE_DIR/<D>/margo.order (hybrid_search)
OUT_DIR=results/engine_order
PY=python
WORKERS=8
SHARES=0,0.1,0.25,0.5,0.75,0.9,1

# Engine side, per budget D with MARGO's share rho under the engine's order (D = 833: 0.25, D = 4167: 0.5), on the
# validation schedule of the make_schedule.py line below (run that line first):
#   python engine/checks/stage_hybrid.py --sets HYPER_DIR/w3/val_Q.npz --schedule OUT_DIR/schedule_w3_val.npz \
#     --key val --queries <qx8.npy> --train-unions HYPER_DIR/w3/train0_U.npz --records 1000000 --pages 83334 \
#     --budget D --arm MARGO LAYOUT_DIR/w3/MARGO_B12.npy rho --no-reference --out ENGINE_DIR/D
#   HYB_ORDER_LOG=1 <hybrid_order tree>/build_8192/apps/hybrid_search int8 l2 <MARGO 8 KB prefix> \
#     ENGINE_DIR/D/val_q.bin ENGINE_DIR/D/val_wptr.u64 ENGINE_DIR/D/val_wstep.u32 500 4 10 12 \
#     ENGINE_DIR/D/hot_MARGO.u32 <lru_pages of val_info.json> ENGINE_DIR/D/margo 0 <warm-up file of the timed runs> \
#     frontier
# The engine's page reads are the page_reads of its summary line.

cd "$(dirname "$0")/.."
mkdir -p "$OUT_DIR"
$PY replay/make_schedule.py w3 --split val --sets "$HYPER_DIR/w3/val_Q.npz" --splits "$DATA_DIR/w3/splits.npz" \
  --calls "$DATA_DIR/w3/calls.jsonl.gz" --drop "$DATA_DIR/w3/drop.json" --out "$OUT_DIR/schedule_w3_val.npz"

L=$LAYOUT_DIR/w3
# budget, hot pages of the cache-aware layouts of Table III (MARGO's lockstep share), hot pages floor(rho * D) of the
# P05-C refined for MARGO's share under the engine's order
for bh in "833 416 208" "4167 3750 2083"; do
  set -- $bh; b=$1; h=$2; he=$3; o=$OUT_DIR/w3-1m_$b
  # one replay set per logged search round, in the engine's order
  $PY replay/engine_order.py --sets "$HYPER_DIR/w3/val_Q.npz" --schedule "$OUT_DIR/schedule_w3_val.npz" \
    --order-log "$ENGINE_DIR/$b/margo.order" --call-map "$ENGINE_DIR/$b/val_lock.npy" \
    --out-sets "$o.rounds_Q.npz" --out-schedule "$o.rounds_schedule.npz"
  # every candidate at every share in that order; MARGO's row of draw 0 at the logged share is what the engine read
  $PY replay/replay_hybrid.py --sets "$o.rounds_Q.npz" --schedule "$o.rounds_schedule.npz" \
    --train-unions "$HYPER_DIR/w3/train{draw}_U.npz" --pages 83334 --budgets $b --shares $SHARES --workers $WORKERS \
    --layout "MARGO=$L/MARGO_B12.npy" --layout "SET-G=$L/SET-G_r{draw}.npy" --layout "ARC-G=$L/ARC-G_r{draw}.npy" \
    --layout "CLQ-G=$L/CLQ-G_r{draw}.npy" --layout "P05-G=$L/P05-G_r{draw}.npy" \
    --layout "ARC-C=$L/ARC-C_h${h}_r{draw}.npy" --layout "P05-C=$L/P05-C_h${h}_r{draw}.npy" \
    --layout "P05-C-$he=$L/P05-C_h${he}_r{draw}.npy" --out "$o.val.json"
  # the order of the candidates breaks exact ties between two of them
  $PY replay/select_deployment.py "$o.val.json" --candidates "ARC-C,ARC-G,CLQ-G,P05-C,P05-C-$he,P05-G,SET-G" \
    --out "$o.selection.json"
done
