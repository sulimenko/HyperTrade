#!/usr/bin/env bash
set -euo pipefail

exec "$HOME/.codex/ai-pipeline/bin/watch-and-run-tasks.sh" "$@"
