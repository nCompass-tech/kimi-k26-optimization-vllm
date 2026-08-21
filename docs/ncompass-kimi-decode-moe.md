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

## 6. Reproducing

```bash
docker build -f docker/Dockerfile.ncompass -t vllm-ncompass:latest .

# run on both nodes with the args in sections 1 and 2, then:
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
grep -a '\[ncompass transport\]' <engine.log>   # RELAY=1 DWARPS=8 RWARPS=32 RBLK=8192 SLOTTED=1 NOAG=1 B2B=1 PUTSIG=1 IBGDA=1
grep -a 'ncompass MoE:'          <engine.log>   # WORLD=16 E_LOCAL=24 epg=24 linear_placement=True
grep -a 'ncompass caps:'         <engine.log>   # P_CAP=720 R_CAP=11520
```

A run missing any of these is void, not passing.
