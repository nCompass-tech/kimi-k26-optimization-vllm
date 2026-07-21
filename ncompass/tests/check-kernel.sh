#!/usr/bin/env bash
# usage: check-kernel.sh <tag>   verify the standalone kernel compiled and runs on GPU.
set -euo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
IMG=$REGISTRY/vllm-openai:${1:?tag}
docker run --rm -i --gpus '"device=0"' --entrypoint python3 "$IMG" - <<'PY'
import torch, ncompass_ext
x = torch.randn(1024, device="cuda")
y = ncompass_ext.scale(x, 2.0)
assert torch.allclose(y, x * 2.0, atol=1e-5), (y[:3], (x * 2)[:3])
print("KERNEL_OK")
PY
