#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
CHECK_MODE=${CHECK_MODE:-default}
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 14), sys.version'
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -m compileall -q hypertrade run_optimize.py run_benchmarks.py run_dashboard.py streamlit_app.py
"$PYTHON_BIN" -c 'import hypertrade.api, hypertrade.ui.app; print("API/dashboard imports OK")'
"$PYTHON_BIN" -m pytest -q
case "$CHECK_MODE" in
  default) ;;
  benchmark) "$PYTHON_BIN" run_benchmarks.py --n_trials 1 ;;
  *) echo "Unknown CHECK_MODE: $CHECK_MODE" >&2; exit 2 ;;
esac
