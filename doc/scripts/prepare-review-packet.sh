#!/usr/bin/env bash
set -euo pipefail

exec "$HOME/.codex/ai-pipeline/bin/prepare-review-packet.sh" "$@"
