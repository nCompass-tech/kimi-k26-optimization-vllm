#!/usr/bin/env bash
# usage: checkpoint.sh <tag>   retag a local-registry image to Docker Hub ncompass/ and push.
# Requires a prior `docker login` with an nCompass Docker Hub token.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
TAG=${1:?tag}
SRC=$REGISTRY/vllm-openai:$TAG
DST=ncompass/vllm-openai:$TAG
docker pull "$SRC"
docker tag "$SRC" "$DST"
docker push "$DST"
echo "CHECKPOINTED $DST"
