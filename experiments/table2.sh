#!/usr/bin/env bash
# [Paper] Table II: units, sub-queries, |e_q|, |E_b|, overlap, L and recall@10 of the six test workloads.
# Inputs are the hypergraphs and logging summaries that prepare/ writes (prepare/README.md, Section 2).
set -euo pipefail

HYPER_DIR=hyper          # HYPER_DIR/<w>/test_Q.npz, test_U.npz
LOG_DIR=logs             # LOG_DIR/<w>/tune_log_summary.json, LOG_DIR/deep/recall.json
OUT_DIR=results/table2
PY=python

cd "$(dirname "$0")/.."
mkdir -p "$OUT_DIR"
$PY prepare/table2.py \
  --row W3-1M  "$HYPER_DIR/w3"   "$LOG_DIR/w3/tune_log_summary.json" \
  --row W3-21M "$HYPER_DIR/w21"  "$LOG_DIR/w21/tune_log_summary.json" \
  --row W2     "$HYPER_DIR/w2"   "$LOG_DIR/w2/tune_log_summary.json" \
  --row W1     "$HYPER_DIR/w1"   "$LOG_DIR/w1/tune_log_summary.json" \
  --row W4     "$HYPER_DIR/w4"   "$LOG_DIR/w4/tune_log_summary.json" \
  --row DEEP1M "$HYPER_DIR/deep" "$LOG_DIR/deep/tune_log_summary.json" "$LOG_DIR/deep/recall.json" \
  --out "$OUT_DIR/table2.json"
