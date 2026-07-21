#!/usr/bin/env bash
# Start the local registry:2 on .28:5000 (idempotent).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
if docker ps --format '{{.Names}}' | grep -qx registry; then
  echo "registry already up"
else
  docker run -d --restart=always --name registry -p 5000:5000 \
    -v "$REGSTORE:/var/lib/registry" registry:2
  sleep 2
fi
curl -fsS "http://$HEAD:5000/v2/" && echo " registry OK"
