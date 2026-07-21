#!/usr/bin/env bash
# Round-trip: push to the local registry on .28, pull it on .30 over the LAN.
set -euo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
docker pull alpine:3.20
docker tag alpine:3.20 "$REGISTRY/alpine:probe"
docker push "$REGISTRY/alpine:probe"                         # localhost push on .28
on2 "docker pull $REGISTRY/alpine:probe && echo PULL_OK_ON_NODE2"   # pull over LAN on .30
docker rmi "$REGISTRY/alpine:probe" >/dev/null 2>&1 || true
