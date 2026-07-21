# ncompass — Kimi-K2.6 Decode-Opt Tooling (read this first)

This directory lives **inside the vLLM checkout** (`/home/ncompass/dev/vllm`) and holds everything
you need to optimize **Kimi-K2.6 decode performance** on the 2-node cluster: launch scripts,
kernels, and tests. You work in the vLLM source (`../vllm/`), add kernels here
(`ncompass/kernels/`), and drive builds/deploys/benchmarks from `ncompass/scripts/`.

**Goal metrics:** lower per-user **median TPOT** and/or raise **output-token throughput**, WITHOUT
regressing TTFT, stability, or model behavior. Valid only if ≥1 primary metric improves and the
other (plus TTFT/behavior) does not regress.
**Baseline to beat:** **6783 ±16 tok/s**, **median TPOT 85.44 ms**, median TTFT ~259 ms, 0 failed.

---

## Layout
```
/home/ncompass/dev/vllm/          # the vLLM checkout — EDIT vllm/*.py here (branch per experiment)
├── vllm/                         # the package that gets overlaid onto the stock wheel
└── ncompass/                     # <-- you are here (custom tooling, travels with the branch)
    ├── README.md                 # this file
    ├── scripts/                  # launch scripts (run these)
    ├── kernels/                  # custom kernel extensions (add new ones here)
    │   └── ncompass_ext/         # worked example: ncompass_ext.scale(x, f) -> x*f
    ├── docker/Dockerfile         # overlay Dockerfile (don't usually touch)
    └── tests/                    # image / kernel / registry checks
```
The design record (spec, plan, milestone results, review) is in `/home/ncompass/kimi-decode-opt/`.

## Cluster topology
- **`.28` = `192.168.40.28`** — THIS box. DP start-rank 0. Hosts the **router** (`:8081`) and the
  **local docker registry** (`:5000`). Run everything from here.
- **`.30` = `192.168.40.30`** — reachable as **`ssh node1`**. DP start-rank 2. `deploy-both.sh`
  drives it for you — you never SSH manually.
- Both: 8×H100. Serving is TP=4, DP=4 (2 local/node), expert-parallel, `--max-model-len 131072`,
  `DecodeBenchConnector` (dummy KV), `FULL_DECODE_ONLY` cudagraphs. **Do not change this config** —
  deploy derives it verbatim from the live systemd unit.

## Environment quirks you MUST know (learned the hard way)
1. **Docker works via a transparent shim.** This login predates docker-group membership, so
   `~/.local/bin/docker` (on PATH) re-runs docker under the right group. Call `docker` normally;
   `scripts/lib.sh` keeps the shim on PATH — **source it** in any new script.
2. **The image ENTRYPOINT is `vllm`.** To run Python in an image, override it:
   `docker run --rm -i --entrypoint python3 <img> - <<'PY' … PY` (`-i` needed for heredoc stdin).
   `scripts/micro.sh` already does this.
3. **`build-push.sh` overlays the checkout's CURRENT working tree.** Confirm `git branch --show-current`
   in the checkout is the branch you intend BEFORE building (a wrong-branch rebuild silently ships
   wrong code — it happened once, caught only by the negative control). Every image is labeled;
   verify with `docker inspect --format '{{.Config.Labels}}' <img> | tr ',' '\n' | grep ncompass`.
4. **Sudo is narrow.** Only `systemctl {restart,stop,start,is-active} inference` + `daemon-reload`
   are NOPASSWD. Anything else (host packages, `/etc`) needs a human. Docker needs no sudo.
5. **Python tooling:** vLLM's own `AGENTS.md` mandates `uv`/`.venv` for *building vLLM itself*. Our
   flow does NOT build vLLM from source — it overlays the prebuilt wheel — so these scripts use the
   image's interpreter inside containers, not a host venv. If you ever rebuild `_C` from source
   (see invariants), follow vLLM's `AGENTS.md`.

## The loop (edit → build → deploy → measure)
Run from `/home/ncompass/dev/vllm/ncompass`:
```bash
# 1) EDIT: in the checkout, one branch per experiment
cd /home/ncompass/dev/vllm && git switch -c ncompass/<experiment>   # edit vllm/*.py
#    new kernel? add ncompass/kernels/<name>/ (see kernels/ncompass_ext for the pattern)
cd ncompass

# 2) BUILD + PUSH to the local registry (add --with-kernel to compile ncompass/kernels)
bash scripts/build-push.sh opt-01-<shortsha> [--with-kernel]

# 3) DEPLOY to BOTH nodes (restarts engines, digest-verifies health; ~5 min engine load)
bash scripts/deploy-both.sh opt-01-<shortsha>

# 4) MEASURE: warm-up + 2 identical measured runs, asserted vs baseline (~30 min)
bash scripts/bench.sh opt-01-<shortsha>
#    results -> ncompass/runs/opt-01-<shortsha>-run{1,2}.txt
```
Fast checks (no deploy): `bash scripts/micro.sh {smoke|marker|serve} <tag>`.
Revert to stock deploy: `bash scripts/revert-stock.sh` (interactive sudo).

## Invariants (don't break these)
- The stock `vllm/vllm-openai:v0.25.0` wheel (upstream commit **`dd10e03f95f94edbea1975c67ace3a35ec9a8a40`**)
  is kept **bit-identical**. Python changes are `COPY`-overlaid; vendor `.so`
  (`_C_stable_libtorch`, `_moe_C_stable_libtorch`, flashmla, …) are stripped from the overlay so
  they're never clobbered. A new kernel is a **separate** compiled extension (`--with-kernel`),
  routed to from Python under a new op name.
- Modifying a kernel *inside* the monolithic `_C` (not reachable as an overridable op) requires
  rebuilding `_C` from source — NOT wired up here, and it REQUIRES a **parity gate**: build the
  baseline commit unchanged from source and confirm it reproduces 6783/85.44 before trusting any
  delta (design spec §5).
- `assert_baseline.py` requires 6000 successful requests + concurrency 600 — a partial run can't
  masquerade as valid. Benchmark parameters are FIXED across measured runs.

## Validating an optimization
Deploy → `bench.sh` (2 runs) → compare to baseline. Report both runs' median TPOT + throughput,
median TTFT (~259 ms), 0 failed, and the exact image tag/commit. A change that improves a metric by
altering the workload, reducing functionality, or adding instability is NOT valid.

## Scripts (`ncompass/scripts/`, all source `lib.sh`)
| script | purpose |
|---|---|
| `preflight.sh` | verify env ready (docker both nodes, sudo, registry, dirs, engines down/up) |
| `registry-up.sh` | start local `registry:2` on `.28:5000` (idempotent) |
| `build-push.sh <tag> [--with-kernel]` | overlay checkout (+kernels) → image → push |
| `deploy-both.sh <tag>` | pull + restart engines on both nodes; digest-verified health |
| `health.sh [--wait] [img]` | engines + router health; asserts running image == pulled digest |
| `bench.sh <label>` | warm-up + 2 measured runs + assert vs baseline (writes `runs/`) |
| `micro.sh {smoke\|marker\|serve} <tag>` | fast import/marker checks, or single-GPU qwen serve |
| `assert_baseline.py <file>` | PASS/FAIL a run vs baseline tolerances |
| `checkpoint.sh <tag>` | (needs `docker login`) push a milestone image to Docker Hub `ncompass/` |
| `revert-stock.sh` | restore stock systemd deploy (interactive sudo) |
| `bootstrap-node.sh` / `root-setup-node.sh` | one-time per-node root setup (already run; re-run bootstrap to tighten sudoers) |

## Current state (handoff)
- Clean `opt-00-baseline` deployed and HEALTHY on both nodes; registry tags: `opt-00-baseline`,
  `opt-test-marker`, `opt-kernel-smoke`.
- Checkout on `ncompass/decode-opt` (clean). Open follow-ups (need a human): tighten live sudoers
  (`sudo bash scripts/bootstrap-node.sh` per node); `docker login` for Docker Hub checkpoints.
