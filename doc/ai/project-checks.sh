#!/usr/bin/env bash
set -euo pipefail

CHECK_MODE=${CHECK_MODE:-default}
PYTHON_BIN=${PYTHON_BIN:-python}

echo "project checks mode=$CHECK_MODE python=$PYTHON_BIN"
"$PYTHON_BIN" -m compileall -q hypertrade

if [ "$CHECK_MODE" = "benchmark" ]; then
  "$PYTHON_BIN" run_benchmarks.py --n_trials 1
fi
