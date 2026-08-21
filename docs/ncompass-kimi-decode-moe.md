# nCompass Kimi-K2.6 decode MoE — deployment contract

One flag turns this on:

```
VLLM_NCOMPASS_KIMI_DECODE_MOE=1
```

Off (the default) is byte-identical stock vLLM. On selects the device-side all-to-all MoE
transport — slot-addressed dispatch, fused put+signal, rail relay, IBGDA, G=1, shape-derived
Marlin M-tile. Everything it selects is an implementation detail of one validated
configuration, not a knob; the values live in `ncompass_moe_config.py`.

**The flag governs code, not provisioning.** It cannot switch on the three things below,
and the engine will not reach the measured numbers without them.

## 1. Serve arguments

EPLB is **required**, not optional — see §4.

```
--enable-eplb
--eplb-config.use_async=false
--eplb-config.communicator=pynccl
--eplb-config.step_interval=500
--eplb-config.window_size=200
--eplb-config.num_redundant_experts=0
```

`use_async=false` avoids async-EPLB's NCCL/NVSHMEM multi-stream conflict. `communicator=pynccl`
is required because vLLM otherwise auto-selects `nixl`, which fails on this cluster with
"NIXL EPLB init failed: buffers".

Also required, from the reference configuration:

```
-cc.mode=VLLM_COMPILE
-cc.cudagraph_mode=FULL_DECODE_ONLY
-cc.splitting_ops=["vllm::unified_kv_cache_update","vllm::unified_mla_kv_cache_update"]
-cc.pass_config.fuse_allreduce_rms=false
--disable-custom-all-reduce
--max-num-batched-tokens 360
--enable-expert-parallel
--tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2
--decode-context-parallel-size 4
```

**Startup validation.** With the flag set, the engine checks `VLLM_ALLREDUCE_USE_SYMM_MEM`,
`pass_config.fuse_allreduce_rms` and `disable_custom_all_reduce`, and refuses to start with
a message naming what is wrong. It validates rather than overrides: silently ignoring an
explicit `fuse_allreduce_rms=true` would hand back a working engine that is not the one you
configured. Unvalidated, each of these surfaces as an illegal memory access during capture,
far from its cause.

**Deviation to disclose:** the shipped deployment omits
`--compilation_config.pass_config.fuse_allreduce_rms true`, which the reference configuration
specifies. With that pass on, `FULL_DECODE_ONLY` capture dies with a deterministic illegal
memory access (the pass collides with symm_mem at the 146 boundary). Disabling it is permitted
under "Allowed Changes → Compilation settings", but it means the in-house 6925 tok/s stock
anchor was measured with the pass **off** and is not the same baseline as the reference
6783 ± 16. Quote which baseline you mean.

## 2. Container environment

```
VLLM_ALLREDUCE_USE_SYMM_MEM=0        # required; the engine validates this
NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_3:1,...,mlx5_9:1
NVSHMEM_IB_ADDR_FAMILY=AF_INET
NVSHMEM_IB_ADDR_RANGE=192.168.40.0/24
IBGDA_NUM_RC=1
```

**No CAP variables.** `R_CAP`/`P_CAP` are derived from the running config:

```
P_CAP = ceil(max_num_batched_tokens / TP) * topk      # ceil(360/4)*8 = 720
R_CAP = EP_WORLD * P_CAP                              # 16*720        = 11520
```

There is no CAP environment variable. The values are derived and asserted at startup.

`R_CAP` is derived **from** `P_CAP`, not independently. The old launcher used
`DP * max_batch * topk`, which equals `EP_WORLD * P_CAP` only when TP divides
`max_num_batched_tokens`. It does at the shipped point (both give 11520), but at e.g. 370 it
gives 11840 against a required 11904 and fails the slotted layout's `R_CAP == WORLD*P_CAP`
invariant at transport construction.

## 3. Image prerequisites

**None external.** Everything ships in the image:

- The IBGDA wrapper is in-tree at `fused_moe/ncompass_ibgda/`. Its compiled object is a build
  output, produced by `build_wrapper.sh` during the image build with that image's own nvcc
  and NVSHMEM, so it cannot drift from what it is nvlinked against.
- There is **no** `sitecustomize` hook. The transport sets the NVSHMEM backend itself, and the
  two vLLM users that could claim it first are disabled by sections 1 and 2 above, which the
  engine validates rather than assumes.

```bash
docker build -f docker/Dockerfile.ncompass -t vllm-ncompass:latest .
```

## 4. Why EPLB is not optional

The three levers are not additive; EPLB is the one that unlocks the others.

| build | EPLB | tile 64 | tile 16 |
|---|---|---:|---:|
| stock AgRs MoE | off | 6925.42 | 7337.74 |
| stock AgRs MoE | on | 7069.72 | 8050.68 |
| ours (deva2a) | off | 6992.61 | 7434.93 |
| ours (deva2a) | **on** | 6989.50 | **8411.95** |

The transport is worth **+1.32%** with EPLB off and **+4.49%** with it on; at tile 64 with
EPLB on it is a net **loss** (−1.13%). Turning EPLB off "to be safe" discards the largest
interaction term in the result.

## 5. Reference numbers

Reference arm = this flag + EPLB + tile 16, measured n=8 over 4 boots:

| metric | value |
|---|---|
| output-token throughput | 8368.96 tok/s mean, range 8163.54–8522.02 (sd 131.89, 4.28%) |
| median TPOT | 67.14 ms |
| median ITL | 648.33 ms, range 647.42–649.26 (0.28%) |
| median TTFT | ~197 ms |
| failed requests | 0 of 48,000 |

**Gate on median ITL, not throughput.** Throughput is wall-time-coupled under EPLB
(`corr(tok/s, duration) = −1.000`) and its run spread is wider than the gap between arms, so
it cannot reliably distinguish this configuration from a partial revert to stock
(8046–8055 tok/s). Median ITL separates them by 7.4% against a 0.28% spread.
Concretely, a pass needs **all** of: median ITL <= 660 ms, median TPOT <= 69 ms, throughput
>= 8140 tok/s, median TTFT <= 320 ms, and **0** failed requests out of 6000.

## 6. Launching under systemd

The reference configuration is a systemd service, and so is this. Ship **two** units rather
than one parameterised unit: the arms then differ only by what §1 and §2 say they differ by,
`systemctl cat` shows the whole command for audit, and no env-file quoting slip can silently
turn one arm into the other. Each declares `Conflicts=` on the other and on `inference.service`,
so the three can never contend for the GPUs.

Do **not** reuse the existing `inference.service`. Its ExecStart hardcodes the reference
arguments and cannot express §1 or §2.

One per-node file, `/etc/ncompass/kdm.env` — the only thing that differs between nodes:

```ini
KDM_IMAGE=vllm-ncompass:latest
KDM_START_RANK=0        # 0 on 192.168.40.28, 2 on 192.168.40.30
```

`/etc/systemd/system/kdm-on.service`. ExecStart is the task spec's Inference Engine Command
with §1 and §2 applied; single-quote `--kv-transfer-config` and `-cc.splitting_ops` so systemd
keeps each as one argv token:

```ini
[Unit]
Description=Kimi-K2.6 decode MoE benchmark arm on
After=docker.service network-online.target
Wants=network-online.target
Requires=docker.service
Conflicts=inference.service kdm-off.service

[Service]
Type=exec
User=ubi
Restart=no
EnvironmentFile=/etc/ncompass/kdm.env
ExecStartPre=-/usr/bin/docker rm -f kdm
ExecStart=/usr/bin/docker run --rm --name kdm \
--privileged \
--env-file /data/srv/etc/inference.env \
--network=host --ipc=host --pid=host \
--ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --ulimit nofile=1048576:1048576 \
--cap-add=IPC_LOCK --cap-add=SYS_NICE \
--device=/dev/infiniband --gpus all \
-v /data/srv/cache:/root/.cache -v /data/srv/models:/models:ro \
-e VLLM_NCOMPASS_KIMI_DECODE_MOE=1 \
-e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
-e IBGDA_NUM_RC=1 \
${KDM_IMAGE} \
--model /models/kimi-k2.6/current \
--served-model-name kimi kimi-k2.6 moonshotai/kimi-k2.6 Kimi Kimi-K2.6 moonshotai/Kimi-K2.6 \
--trust-remote-code --host 0.0.0.0 \
--tensor-parallel-size 4 --decode-context-parallel-size 4 \
--data-parallel-size 4 --data-parallel-size-local 2 \
--data-parallel-address 192.168.40.28 --data-parallel-rpc-port 13345 \
--data-parallel-hybrid-lb --enable-expert-parallel \
--max-model-len 131072 --mm-encoder-tp-mode data \
--tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --enable-auto-tool-choice \
--max-num-batched-tokens 360 \
--kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
-cc.cudagraph_mode=FULL_DECODE_ONLY \
--stream-interval 10 \
--data-parallel-start-rank ${KDM_START_RANK} \
--enable-eplb \
--eplb-config.use_async=false \
--eplb-config.communicator=pynccl \
--eplb-config.step_interval=500 \
--eplb-config.window_size=200 \
--eplb-config.num_redundant_experts=0 \
-cc.mode=VLLM_COMPILE \
'-cc.splitting_ops=["vllm::unified_kv_cache_update","vllm::unified_mla_kv_cache_update"]' \
-cc.pass_config.fuse_allreduce_rms=false \
--disable-custom-all-reduce
ExecStop=/usr/bin/docker stop -t 60 kdm
TimeoutStopSec=90
TimeoutStartSec=1800
StandardOutput=journal
StandardError=journal
```

`/etc/systemd/system/kdm-off.service` is the same file with the flag off — i.e. stock vLLM
under the reference configuration, which is what the spec baseline measures. Three changes:
drop the three `-e` lines; drop the ten §1 arguments after `--data-parallel-start-rank`; and
restore the reference `--compilation_config.pass_config.fuse_allreduce_rms true` that §1
replaces. Also flip `Conflicts=` to name `kdm-on.service`.

Install on **both** nodes, then `systemctl daemon-reload`. Verify the quoting took before
trusting a run — systemd's own parser is the authority:

```bash
systemctl show -p ExecStart kdm-on.service | tr ';' '\n' | sed -n 's/^ *argv\[\]=//p'
```

The two arms must differ by exactly 14 argv tokens (+6 for three `-e` flags, −2 for the
dropped fusion flag, +10 for §1). `${KDM_IMAGE}` and `${KDM_START_RANK}` show unexpanded
here; systemd substitutes them at exec time.

Driving it — `systemctl` needs root, and a rule matching `inference` will not match these:

```bash
sudo systemctl start kdm-on     # on BOTH nodes; kdm-off for the baseline arm
docker logs -f kdm              # NOT journalctl unless the caller is in adm/systemd-journal
```

To run a campaign unattended, add `/etc/sudoers.d/ncompass-kdm` (mode 0440) with one rule
per line — long comma-separated lists wrap when pasted and break the parser — for
`start`/`stop`/`is-active` on `kdm-on`, `kdm-off` and their `.service` forms. Validate with
`visudo -cf` before installing. `sudoers` is per host: install it on both nodes, since
`ssh node2 'sudo systemctl start kdm-on'` is governed by node 2's rules.

**Readiness is two gates, not one.** `:8000/health` returning 200 means the *engine* is up.
The router keeps a separate circuit breaker and trips it while engines restart; its health
checker runs on a 60 s interval. Benchmarking in that window returns 6000 × `HTTP 503
No available workers (all circuits open or unhealthy)` and a 0.00 tok/s report that looks
like a catastrophic regression rather than a harness error. Poll a real routed completion
until it answers before starting any measured run:

```bash
until [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 \
  http://<router>:8081/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"moonshotai/Kimi-K2.6","messages":[{"role":"user","content":"hi"}],"max_tokens":8}')" \
  = 200 ]; do sleep 15; done
```

## 7. Reproducing

```bash
docker build -f docker/Dockerfile.ncompass -t vllm-ncompass:latest .

# start both nodes per section 6, wait for both readiness gates, then:
./vllm-bench --backend openai-chat --model moonshotai/Kimi-K2.6 \
  --base-url http://<router>:8081 --max-concurrency 600 --num-prompts 6000 \
  --dataset-name random --random-input-len 5120 --random-output-len 1280 \
  --random-range-ratio 0.5 --random-prefix-len 4096 --ignore-eos --trust-remote-code
```

`vllm-bench` is the official `vllm-project/vllm-bench` **v0.1.0** release
(`sha256 e2e246df...20716c`). Pin it: if the driver changes, the reference numbers above stop
being comparable.

Every gate in this system fails safe-to-slow: a misconfigured engine serves stock throughput
happily rather than erroring, so a green benchmark is **not** evidence the optimization ran.
Assert it positively, from inside the running container:

```bash
docker logs kdm 2>&1 | grep -a '\[ncompass transport\]'  # RELAY=1 DWARPS=8 RWARPS=32 RBLK=8192 SLOTTED=1 NOAG=1 B2B=1 PUTSIG=1 IBGDA=1
docker logs kdm 2>&1 | grep -a 'ncompass MoE:'           # WORLD=16 E_LOCAL=24 epg=24 linear_placement=True
docker logs kdm 2>&1 | grep -a 'ncompass caps:'          # P_CAP=720 R_CAP=11520
```

Read these with `docker logs`, not `journalctl`: the units log to the journal, but the account
driving the benchmark is typically in `docker` and not in `adm`, so `journalctl -u kdm-on`
returns nothing and reads as a missing marker rather than a permissions problem.

A run missing any of these is void, not passing.
