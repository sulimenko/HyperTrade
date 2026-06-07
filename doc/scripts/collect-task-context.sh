#!/usr/bin/env bash
set -euo pipefail

exec "$HOME/.codex/ai-pipeline/bin/collect-task-context.sh" "$@"
