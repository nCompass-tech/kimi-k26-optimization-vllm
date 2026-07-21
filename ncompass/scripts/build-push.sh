#!/usr/bin/env bash
# usage: build-push.sh <tag> [--with-kernel]
# Overlays the vLLM checkout (this repo's vllm/ package) onto the stock wheel, optionally compiles
# kernels from ncompass/kernels, tags for the local registry, and pushes.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
TAG=${1:?tag required}
WITH_KERNEL=${2:-}
IMG=$REGISTRY/vllm-openai:$TAG
NCROOT=$(cd "$(dirname "$0")/.." && pwd)     # /home/ncompass/dev/vllm/ncompass

BCTX=$(mktemp -d)
trap 'rm -rf "$BCTX"' EXIT
cp "$NCROOT/docker/Dockerfile" "$BCTX/Dockerfile"
mkdir -p "$BCTX/kernels"; : > "$BCTX/kernels/.keep"
if [ "$WITH_KERNEL" = "--with-kernel" ]; then
  cp -a "$NCROOT/kernels/." "$BCTX/kernels/"
  echo "including kernels from ncompass/kernels"
fi
mkdir -p "$BCTX/dev/vllm"
# only the vLLM package (no .git/tests/docs/ncompass) — hardlink for speed, fall back to copy
cp -al "$CHECKOUT/vllm" "$BCTX/dev/vllm/vllm" 2>/dev/null || cp -a "$CHECKOUT/vllm" "$BCTX/dev/vllm/vllm"
# NEVER overlay locally-built kernels or bytecode onto the stock wheel — an in-place build in
# $CHECKOUT would otherwise clobber the vendor .so and every gate would still pass. Strip them.
find "$BCTX/dev/vllm/vllm" \( -name '*.so' -o -name '__pycache__' \) -exec rm -rf {} + 2>/dev/null || true

# stamp provenance so a mislabeled image (e.g. baseline accidentally built from a feature
# branch) is visible via `docker inspect --format '{{.Config.Labels}}'`
SRC_BRANCH=$(git -C "$CHECKOUT" branch --show-current 2>/dev/null || echo detached)
SRC_COMMIT=$(git -C "$CHECKOUT" rev-parse --short HEAD 2>/dev/null || echo unknown)
DOCKER_BUILDKIT=1 docker build \
  --label "ncompass.src_branch=$SRC_BRANCH" \
  --label "ncompass.src_commit=$SRC_COMMIT" \
  --label "ncompass.with_kernel=$([ "$WITH_KERNEL" = "--with-kernel" ] && echo yes || echo no)" \
  -f "$BCTX/Dockerfile" -t "$IMG" "$BCTX"
docker push "$IMG"
echo "BUILT_AND_PUSHED $IMG  (src: $SRC_BRANCH @ $SRC_COMMIT)"
