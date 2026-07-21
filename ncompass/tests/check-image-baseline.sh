#!/usr/bin/env bash
# Verify opt-00-baseline is a clean overlay: correct version/commit, prebuilt .so intact.
set -euo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
IMG=$REGISTRY/vllm-openai:opt-00-baseline
docker run --rm -i --entrypoint python3 "$IMG" - <<'PY'
import vllm, os, glob
d = os.path.dirname(vllm.__file__)
assert vllm.__version__ == "0.25.0", vllm.__version__
sos = [os.path.basename(p) for p in glob.glob(os.path.join(d, "*.so"))]
for need in ("_C_stable_libtorch", "_moe_C_stable_libtorch"):
    assert any(need in s for s in sos), f"missing prebuilt {need}: {sos}"
from vllm._version import __commit_id__
assert __commit_id__ == "gdd10e03f9", __commit_id__
print("IMAGE_BASELINE_OK", vllm.__version__, __commit_id__, len(sos), "so files")
PY
