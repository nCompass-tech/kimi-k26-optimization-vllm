# syntax=docker/dockerfile:1.7
FROM vllm/vllm-openai:v0.25.0

# tools needed only to compile a standalone kernel extension (Python-only builds don't use them)
RUN apt-get update && apt-get install -y --no-install-recommends cmake ccache && rm -rf /var/lib/apt/lists/*

ARG VLLM_SITE=/usr/local/lib/python3.12/dist-packages/vllm

# 1) Overlay the edited vLLM source onto the stock wheel. COPY merges: edited .py overwrite the
#    installed ones; vendor .so are stripped from the context (see build-push.sh) so the prebuilt
#    kernels are preserved. Preserve the wheel's generated _version.py; drop stale bytecode.
RUN cp "$VLLM_SITE/_version.py" /tmp/_version.py
COPY dev/vllm/vllm/ ${VLLM_SITE}/
RUN cp /tmp/_version.py "$VLLM_SITE/_version.py" && rm /tmp/_version.py \
 && find "$VLLM_SITE" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

# 2) (optional) build standalone kernel extension(s) from ncompass/kernels. No-op when absent.
COPY kernels/ /opt/ncompass/kernels/
RUN --mount=type=cache,target=/ccache \
    if [ -f /opt/ncompass/kernels/ncompass_ext/setup.py ]; then \
      export CCACHE_DIR=/ccache PATH="/usr/lib/ccache:$PATH"; \
      pip install --no-build-isolation /opt/ncompass/kernels/ncompass_ext; \
    else echo "no kernel extension to build"; fi
