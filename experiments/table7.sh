#!/usr/bin/env bash
# [Paper] Table VII: saving over MARGO of HyperPage's candidates in the W3 columns (rows ARC-G, P05-G, ARC-C, P05-C,
# ARC-CF, HEAT's objective = HEAT-res, MARGO on uncached records = MARGO-uncached); bold marks the deployed pick,
# which select_deployment.py recorded. Reads the replays of table3.sh, which must run first.
set -euo pipefail

OUT_DIR=results/table3   # output directory of table3.sh
PY=python

cd "$(dirname "$0")/.."
for column in w3-1m_833 w3-1m_4167 w3-1m_20834 w3-21m_17513 w3-21m_87564 w3-21m_262144 w3-21m_437820; do
  echo "== $column"
  $PY replay/saving_table.py --selection "$OUT_DIR/$column.selection.json" --test "$OUT_DIR/$column.test.json" |
    grep -E "MARGO reads|ARC-G |P05-G |ARC-C |P05-C |ARC-CF |HEAT-res |MARGO-uncached |HyperPage"
  $PY -c "import json, sys; print('pick per draw:', [(p['layout'], p['share']) for v in json.load(open(sys.argv[1]))['deployment'].values() for p in v])" \
    "$OUT_DIR/$column.selection.json"
done
