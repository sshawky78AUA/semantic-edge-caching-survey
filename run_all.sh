#!/usr/bin/env bash
# Regenerates the figures of Section IX.
# Usage: ./run_all.sh            (figures from the stored results)
#        ./run_all.sh --rerun    (also reruns every simulation)
set -euo pipefail
cd "$(dirname "$0")/sim"
if [ "${1:-}" = "--rerun" ]; then
  python3 experiments2.py validate e1 e1r e2 e3 e4 e5
fi
python3 figs2.py e1 e1r e2 e3 e4 e5
echo "Figures written to figs/"
