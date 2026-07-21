#!/usr/bin/env bash
# One-time LEAST-PRIVILEGE root setup per node for the Kimi decode-opt dev flow.
# Run as root on EACH node:  sudo bash root-setup-node.sh
# Idempotent. Does NOT grant standing sudo. It does exactly two things:
#   1) systemd drop-in so `inference` launches ${INFERENCE_IMAGE} at this node's DP rank
#      (derived verbatim from the live stock unit, so every serving flag is preserved)
#   2) docker insecure-registries for 192.168.40.28:5000, then restart docker
# After this, the narrow `systemctl … inference` NOPASSWD rule covers the whole deploy loop.
set -euo pipefail
REG="192.168.40.28:5000"

# --- 1) systemd dev drop-in ---
EXEC=$(systemctl cat inference | awk '
  /^ExecStart=\/usr\/bin\/docker run/ {c=1}
  c {print}
  c && $0 !~ /\\$/ {exit}')
RANK=$(grep -oP 'data-parallel-start-rank \K[0-9]+' <<<"$EXEC")
EXEC=$(sed -E \
  -e 's#vllm/vllm-openai:v0\.25\.0#${INFERENCE_IMAGE}#' \
  -e 's#--data-parallel-start-rank [0-9]+#--data-parallel-start-rank ${INFERENCE_START_RANK}#' \
  <<<"$EXEC")
# Fail loudly if the stock unit's image/rank strings didn't match — otherwise the drop-in would
# silently keep launching the hardcoded stock image while deploy-both.sh believes it changed it.
grep -q '${INFERENCE_IMAGE}' <<<"$EXEC" || { echo "ERROR: image substitution failed"; exit 1; }
grep -q '${INFERENCE_START_RANK}' <<<"$EXEC" || { echo "ERROR: rank substitution failed"; exit 1; }
mkdir -p /etc/ncompass /etc/systemd/system/inference.service.d
printf 'INFERENCE_IMAGE=%s\nINFERENCE_START_RANK=%s\n' \
  "$REG/vllm-openai:opt-00-baseline" "$RANK" > /etc/ncompass/inference-dev.env
chown ncompass /etc/ncompass/inference-dev.env
chmod 644 /etc/ncompass/inference-dev.env
{ echo "[Service]"
  echo "EnvironmentFile=/etc/ncompass/inference-dev.env"
  echo "ExecStart="
  echo "$EXEC"
} > /etc/systemd/system/inference.service.d/dev.conf
systemctl daemon-reload

# --- 2) docker insecure-registries (merge-safe) + restart ---
python3 - "$REG" <<'PY'
import json, os, sys, shutil
reg = sys.argv[1]; p = "/etc/docker/daemon.json"; d = {}
if os.path.exists(p):
    try:
        d = json.load(open(p))
    except Exception as e:
        sys.exit(f"refusing to rewrite unparseable {p} ({e}) — fix it by hand")
    shutil.copy(p, p + ".bak")     # back up before rewriting
lst = d.get("insecure-registries", [])
if reg not in lst: lst.append(reg)
d["insecure-registries"] = lst
json.dump(d, open(p, "w"), indent=2)
PY
systemctl restart docker
sleep 3

echo "ROOT_SETUP_OK on $(hostname): rank=$RANK, dev drop-in installed, insecure-registries set, docker restarted"
