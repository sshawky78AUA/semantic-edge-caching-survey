#!/usr/bin/env bash
# Reproduces the simulation results, figures, and table values of the article and its supplement.
#
# Usage:
#   ./run_all.sh               figures and table values from the stored results (seconds)
#   ./run_all.sh --rerun       reruns every simulation of the article and the supplement, then
#                              rebuilds the figures and tables (about 23,000 runs, roughly 3 CPU-hours)
#   ./run_all.sh --quick       smoke test: every simulation with 3 seeds per point, written to
#                              quick/results and quick/figs, so the stored results are not touched
#   ./run_all.sh --rerun --data  also reruns the checks on public data (download the data first,
#                              see sim/realdata.py)
#
# Environment variables: UCM_WORKERS (processes, default all cores), UCM_RUNS (seeds per point),
# UCM_RES (results folder, default sim/), UCM_FIGS (figures folder, default figs/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/sim"
PY="${PYTHON:-python3}"

RERUN=0; DATA=0
for a in "$@"; do
  case "$a" in
    --rerun) RERUN=1 ;;
    --quick) RERUN=1; export UCM_RUNS=3; export UCM_RES="$ROOT/quick/results"; export UCM_FIGS="$ROOT/quick/figs/" ;;
    --data) DATA=1 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done

if [ "$RERUN" = 1 ]; then
  echo "== Section IX experiments: validation, E1 to E5, E1r (30 seeds per point)"
  $PY experiments2.py validate e1 e1r e2 e3 e4 e5
  echo "== Control experiments (supplement)"
  $PY experiments3.py e1p
  $PY experiments3.py e4s
  for p in 0 1 2; do $PY experiments3.py e2r "$p"; done
  for e in e1c e1t e1m e3t e5s e5sb; do $PY experiments4.py "$e"; done
  $PY experiments4.py e5t 0.9
  $PY experiments4.py e5t 0.8
  echo "== E5b (EWMA low level) and E2f (reward grid of step 0.05)"
  $PY experiments5.py e5b e2f
  echo "== E6r (interaction versus storage per semantic class)"
  $PY experiments6.py e6r
  if [ "$DATA" = 1 ]; then
    echo "== Checks on public data"
    $PY realdata.py trace
    $PY realdata.py sts
    $PY realdata.py sts2
  fi
  if [ -n "${UCM_RES:-}" ]; then
    # results that need external data are copied from the stored ones in quick mode
    for f in res5_trace.json res5_sts.json res6_sts.json; do
      [ -f "$UCM_RES/$f" ] || cp "$ROOT/sim/$f" "$UCM_RES/$f"
    done
  fi
fi

echo "== Figures"
$PY figs2.py e1 e1r e2 e3 e4 e5
echo "== Table values"
$PY tables.py
echo "Figures written to ${UCM_FIGS:-$ROOT/figs/}"
