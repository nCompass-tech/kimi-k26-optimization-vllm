#!/usr/bin/env bash
# Single-node micro rig.
#   micro.sh smoke  <tag>   instant import check (no GPU)
#   micro.sh marker <tag>   print the build marker (no GPU)
#   micro.sh serve  <tag>   single-node qwen3-0.6b server on :8009 (GPU 0) for profiling
set -euo pipefail
source "$(dirname "$0")/lib.sh"
sub=${1:?smoke|marker|serve}; TAG=${2:?tag}; IMG=$REGISTRY/vllm-openai:$TAG
case "$sub" in
  smoke)  docker run --rm --entrypoint python3 "$IMG" -c "import vllm; print('SMOKE_OK', vllm.__version__)";;
  marker) docker run --rm --entrypoint python3 "$IMG" -c "import vllm; print('MARKER=' + getattr(vllm,'NCOMPASS_BUILD','<none>'))";;
  serve)  Q=$(docker run --rm --entrypoint bash "$IMG" -lc 'ls -d /models/qwen3-0.6b/* | head -1')
          docker run --rm --gpus '"device=0"' --network=host "$IMG" \
            --model "$Q" --served-model-name qwen --max-model-len 4096 --port 8009;;
  *) echo "unknown subcommand: $sub"; exit 2;;
esac
