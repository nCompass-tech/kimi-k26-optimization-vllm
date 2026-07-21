#!/usr/bin/env bash
# Verify the vLLM checkout is pinned at the baseline commit and clean.
set -euo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
cd "$CHECKOUT"
test "$(git rev-parse HEAD)" = "$BASELINE_COMMIT" && echo "HEAD ok"
test -f vllm/__init__.py && test -d vllm/distributed/kv_transfer/kv_connector/v1 && echo "tree ok"
git status --porcelain | grep -q . && { echo "DIRTY"; exit 1; } || echo "clean"
