#!/usr/bin/env bash
# Regenerates the figures of Section IX.
# Usage: ./run_all.sh            (figures from the stored results)
#        ./run_all.sh --rerun    (also reruns every simulation)
set -euo pipefail
cd "$(dirname "$0")/sim"
if [ "${1:-}" = "--rerun" ]; then
  python3 experiments2.py validate e1 e1r e2 e3 e4 e5
  # Control experiments of the revision (20 seeds each)
  python3 experiments3.py e1p
  python3 experiments3.py e4s
  for p in 0 1 2; do python3 experiments3.py e2r "$p"; done
  for e in e1c e1t e1m e3t e5s e5sb; do python3 experiments4.py "$e"; done
  python3 experiments4.py e5t 0.9
  python3 experiments4.py e5t 0.8
fi
python3 figs2.py e1 e1r e2 e3 e4 e5
echo "Figures written to figs/"
