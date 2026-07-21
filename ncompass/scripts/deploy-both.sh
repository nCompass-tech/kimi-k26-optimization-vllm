#!/usr/bin/env bash
# usage: deploy-both.sh <tag>   pull image + restart engines on .28 and .30, wait for health
set -euo pipefail
source "$(dirname "$0")/lib.sh"
IMG=$REGISTRY/vllm-openai:${1:?tag required}

echo "pulling $IMG on both nodes…"
docker pull "$IMG" >/dev/null && echo "  pulled on .28"
on2 "docker pull $IMG >/dev/null && echo '  pulled on .30'"

# Rewrite the ncompass-owned env file in place (dir is root-only, so use > not sed -i),
# preserving each node's DP start-rank.
setimg() { # $1 = target (local uses ""; passed via on2 for remote)
  local rank
  rank=$(grep -oP 'INFERENCE_START_RANK=\K.*' /etc/ncompass/inference-dev.env)
  printf 'INFERENCE_IMAGE=%s\nINFERENCE_START_RANK=%s\n' "$IMG" "$rank" > /etc/ncompass/inference-dev.env
}
setimg
on2 "rank=\$(grep -oP 'INFERENCE_START_RANK=\K.*' /etc/ncompass/inference-dev.env); printf 'INFERENCE_IMAGE=%s\nINFERENCE_START_RANK=%s\n' '$IMG' \"\$rank\" > /etc/ncompass/inference-dev.env"

echo "restarting engines on both nodes…"
sudo systemctl daemon-reload && sudo systemctl restart inference
on2 "sudo systemctl daemon-reload && sudo systemctl restart inference"

echo "RESTARTED both engines on $IMG; waiting for health (up to 1200s)…"
bash "$(dirname "$0")/health.sh" --wait "$IMG"   # asserts running image ID == pulled digest on both nodes
