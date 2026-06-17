#!/usr/bin/env bash
# End-to-end: calibrate k_risk -> run the Monte Carlo sweep -> build the HTML report.
# Usage: examples/run_all.sh [OUT_DIR]
set -euo pipefail

OUT="${1:-runs/latest}"
cd "$(dirname "$0")/.."

echo "==> 1/3 calibrate (fit k_risk to the regime bands, §7.2)"
corridor-sim calibrate --config config/default.yaml --out "$OUT"

echo "==> 2/3 run sweep (corridors × turnover × regime × model, §6)"
# Reuse the just-fitted k_risk from the calibrated, resolved config.
corridor-sim run --config "$OUT/config_resolved.yaml" --out "$OUT" --trials 10000 --seed 42 --no-calibrate

echo "==> 3/3 build report"
corridor-sim report --in "$OUT" --out "$OUT/report.html"

echo "Done. Open $OUT/report.html"
