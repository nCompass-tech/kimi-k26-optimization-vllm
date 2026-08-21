# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ncompass: G-way expert-group MoE core (device-initiated all-to-all dispatch/combine
+ Marlin compute, 2-stream overlap).

Layer-agnostic core, extracted from ncompass_moe_layer.ncompass_moe_apply (Task 3 refactor:
behavior-preserving move, no vLLM `layer`/`get_ep_group()` coupling). Split this rank's local
experts into G groups (epg = E_LOCAL/G), route each (token,expert) pair to its owner rank,
run Marlin per group, send back, weighted scatter-add to origin. Groups pipeline on 2 streams:
dispatch(g+1) overlaps compute(g); at the shipped G=1 that reduces to a single pass.

Dispatch/combine go through ncompass_transport (device-initiated NVSHMEM/IBGDA puts), NOT
`dist.all_to_all_single` -- the transport is what removes the host split-list round-trip and
makes the region CUDA-graph capturable. `comm` is still a torch.distributed process group,
used for the histogram all-gather and passed through to transport construction.

Marlin is NOT grid-capped: the sms_override lever was measured and rejected, and the tile is
chosen by `_select_block_size_m` from the true routing shape instead.

The vLLM shell (ncompass_moe_layer.py) supplies `comm`/`ep_rank`/`ep_world`, builds/caches
`operands` via build_marlin_operands(), and handles the shared-expert add -- none of that
lives here.

Assumes LINEAR expert placement (rank r owns globals [r*E_LOCAL,(r+1)*E_LOCAL)); the shell
sanity-checks + logs this on first call.
"""

import math

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.masked_activation import (
    fused_combine_epilogue, masked_silu_and_mul, slotted_silu_and_mul)
from vllm.model_executor.layers.fused_moe import ncompass_moe_config as _cfg
from vllm.model_executor.layers.fused_moe.ncompass_routing_meta import build_group_meta_hist

# Expert groups. G=1 is the shipped configuration; the G=2 pipeline is retained only
# because the code is written group-generically, not because it is used.
G = _cfg.EXPERT_GROUPS
# Skew margin on rows-per-expert: size the tile for the BUSIEST expert, not the average.
# An expert that overflows its tile costs a whole extra M-tile, which re-streams that
# expert's weights from DRAM. Validated against a 96-point measured sweep: 1.0-1.1
# reproduce the optimal tile at 13/16 shapes; >=1.2 degrades to 9/16.
_TILE_SKEW = 1.1
_MARLIN_TILES = (8, 16, 32, 48, 64)
# ENABLED by default (2026-08-19). Set NCOMPASS_MARLIN_TILE_AUTO=0 to fall back to the
# constant below.
#
# This selector was previously gated OFF on a measured "2.0% regression" (8199.87 vs
# 8368.96 tok/s). That attribution was WRONG, and the correction is worth recording
# because the trap is easy to fall into again:
#
#   1. The selector emits tile 16 at every graph size that actually replays. Measured
#      in-engine: T=40 -> r_max 14.67 -> 16, and T=38 -> 13.93 -> 16. A 16-rank trace
#      recovers those T values from 16 independent kernel grids (the decisive one is
#      _k_fused_combine_epilogue's 41x7 grid = the [T+1, 7168] buffer). So the derived
#      arm and the hardcoded arm run the SAME TILE in steady state and cannot differ
#      because of it.
#   2. The "derived tile = 48" reading came from CAPTURE, not replay. Capture walks the
#      whole size ladder up to the SP-sharded cap T=90, and T in [88,130] is exactly the
#      window that yields 48 -- so the largest captured size truthfully logs 48 while
#      every executed size logs 16.
#   3. The throughput gap is wall-time variance, not decode work. Over 12 runs
#      (8 hardcoded / 4 derived) MEDIAN ITL -- which excludes stall time -- is
#      648.27 vs 648.34 ms, +0.01%, t=0.21. Throughput (-1.86%, t=-1.98), median TPOT
#      (+2.61%) and benchmark duration (+1.89%, t=1.99) all move together because all
#      three carry EPLB rearrangement stalls; corr(medTPOT, duration) = +0.77.
#      Median TPOT is NOT stall-robust: vLLM computes it per request as
#      (E2E - TTFT)/(tokens-1), so a rearrangement that stalls all 600 in-flight
#      requests shifts the whole distribution. Median ITL is the metric that isolates
#      the work this tile controls, and it shows nothing.
#
# CAVEAT worth carrying: _TILE_SKEW = 1.1 is fitted on the isolated harness, not
# measured in production. At T=40 the mean is 13.33 rows/expert and the tile-16/32
# boundary sits at r_max = 16, so the margin is only 9.5%. Per-layer Marlin cost spans
# 104-227 us across the 60 layers, so real skew is large and layer-dependent; if a
# layer's true r_max
# crosses 16 the graph-wide constant holds it at 16 anyway. Grounding r_max in
# moe_align_block_size's own per-expert histogram (its CUDA kernel already computes it;
# the Python wrapper discards it) would remove this assumption.
#
# Fallback tile when AUTO is off: 16, the measured optimum at the current operating
# point (8368.96 tok/s over 8 runs / 4 boots).
_MARLIN_TILE_DEFAULT = 16
_TILE_AUTO = True
# E1 evidence collection; see profiling-specs/2026-08-19-marlin-tile-runtime-selector
# -row-count.md.
# Default off; the shipped path is byte-identical with it unset.


def _select_block_size_m(tokens: int, topk: int, e_local: int, ep_world: int) -> int:
    """Pick Marlin's M-tile from the routing shape. -1 defers to vLLM's heuristic.

    Rule, derived and validated in
    ncompass/profiling-reports/2026-08-19-marlin-tile-attribution.md: optimal tile
    is the SMALLEST T minimising ceil(r_max / T), the M-tiles each expert needs.
    Two regimes meet at r_max == T:

      r_max <= T   every candidate tile needs one M-tile, so an expert's weights stream
                   from DRAM once regardless of tile. Padding is then the only cost and
                   shrinking is free -- measured: DRAM reads move 0.28% while MMA work
                   moves 3.84x, and the kernel runs 50% faster.
      r_max >  T   shrinking adds M-tiles, each re-reading that expert's weights, so
                   bandwidth dominates -- measured at 85 rows/expert: tile 16 costs +48%
                   DRAM and runs 18.4% SLOWER than tile 64.

    So halving the tile pays iff it does not raise ceil(r_max/T), i.e. iff r_max <= T/2.

    Mechanism (NCU): padding drives it; occupancy is a SECOND-ORDER CONSEQUENCE, not an
    independent effect -- thread_m_blocks=4 forces accumulator registers 32->156, 89% of
    the growth that caps tile 64 at 2 CTAs/SM. Tile 64 cannot be run at 3 blocks/SM.

    This replaces a hardcoded 16, which is optimal ONLY near 12 rows/expert (our current
    decode point) and up to 53% off elsewhere. Against the sweep this rule averages
    0.45% off optimal (worst 4.1%); a fixed 16 averages 23% off.

    Must be a HOST-SIDE CONSTANT -- it fixes sorted_token_ids' shape and the kernel's
    template dispatch, both frozen at CUDA-graph capture. Every input here is a
    host-side shape, so it is decided once per captured graph size, stable at replay.
    """
    # ep_world is validated but does NOT enter the arithmetic; see below.
    if tokens <= 0 or topk <= 0 or e_local <= 0 or ep_world <= 0:
        return -1
    # Rows on ONE local expert. This rank RECEIVES pairs from all ep_world ranks and
    # owns
    # e_local of the ep_world*e_local global experts, so the two ep_world factors
    # CANCEL:
    #   rows_here = (ep_world * tokens * topk) * (e_local / (ep_world * e_local))
    #             = tokens * topk
    # Dividing by ep_world here as well is the bug the first null test caught -- it
    # under-
    # counted 16x and selected tile 8 instead of 16. Cross-check: T=37, topk=8,
    # e_local=24
    # -> 12.3 rows/expert, matching the 12.5 measured in production.
    r_max = (tokens * topk / e_local) * _TILE_SKEW
    if r_max <= 0:
        return -1
    best, best_tiles = -1, None
    for t in _MARLIN_TILES:          # ascending; tile count is non-increasing in t
        n = math.ceil(r_max / t)
        if best_tiles is None or n < best_tiles:
            best_tiles, best = n, t  # smallest tile achieving the minimum tile count
    return best


# R_CAP: when transport is on, size recv_topk_ids to the (static) recv CAP with sentinel
# padding so
# Marlin runs on the full CAP recv buffer (dispatch returns the whole buffer, not
# recv[:R]).
_FUSED_EPI = _cfg.FUSED_EPI   # B2a
_compute_stream = None
_transport = None


def _get_transport(comm, H, dtype, dev, world, rank):
    global _transport
    if _transport is None:
        from vllm.model_executor.layers.fused_moe.ncompass_transport import NcompassMoETransport
        _transport = NcompassMoETransport(comm, H, dtype, dev, world, rank)
    return _transport


def transport_is_initialized() -> bool:
    return _transport is not None


def get_transport_or_none():
    """The live transport instance, or None if it was never built.

    NON-CONSTRUCTING, unlike `_get_transport`: `NcompassMoETransport.__init__` performs a collective symm_mem
    rendezvous, so a caller that only wants to READ transport state must never be the one that
    creates it. Used by the EPLB rearrangement-boundary instrumentation.
    """
    return _transport


def ensure_transport(comm, H, dtype, dev, world, rank):
    """Construct the transport EAGERLY, outside any CUDA-graph capture.

    `NcompassMoETransport.__init__` performs a `torch.distributed._symmetric_memory` rendezvous -- a COLLECTIVE,
    host-side allocation. It is neither capture-safe nor safe to enter on a subset of ranks.

    While our MoE handled every batch, this was harmless: the first call always landed in an eager
    warmup, so the transport existed long before capture. Decode-only routing broke that invariant --
    non-uniform warmup batches go to the stock kernel, so the FIRST construction would happen inside
    capture, where the rendezvous hangs or aborts across ranks. That is what produced the silent
    worker death during memory profiling, with no traceback on any rank.

    Callers route this in during an eager forward so the invariant is restored explicitly instead of
    being an accident of which batch shape happens to arrive first.
    """
    return _get_transport(comm, H, dtype, dev, world, rank)


def _cs():
    global _compute_stream
    if _compute_stream is None:
        _compute_stream = torch.cuda.Stream()
    return _compute_stream


def _slice0(t, a, b):
    return None if t is None else t[a:b].contiguous()


def build_marlin_operands(w13_weight, w2_weight, w1_scale, w2_scale, w1_zp, w2_zp,
                         g_idx1, g_idx2, sort_indices1, sort_indices2,
                         quant_type_id, is_k_full, input_dtype, activation):
    """Static per-group weight slices (packed Marlin, dim0 = local experts) for G groups.

    Layer-agnostic port of the inline slicing formerly in ncompass_moe_apply's
    `_cached_marlin_operands`; caller (vLLM shell) is responsible for caching the result since
    weights never change.
    """
    E = w13_weight.shape[0]
    assert E % G == 0, f"local experts {E} not divisible by G={G}"
    epg = E // G
    slices = []
    for g in range(G):
        a, b = g * epg, (g + 1) * epg
        slices.append({
            "w1": _slice0(w13_weight, a, b), "w2": _slice0(w2_weight, a, b),
            "w1_scale": _slice0(w1_scale, a, b), "w2_scale": _slice0(w2_scale, a, b),
            "w1_zp": _slice0(w1_zp, a, b), "w2_zp": _slice0(w2_zp, a, b),
            "g_idx1": _slice0(g_idx1, a, b), "g_idx2": _slice0(g_idx2, a, b),
            "si1": _slice0(sort_indices1, a, b),
            "si2": _slice0(sort_indices2, a, b),
        })
    return {
        "epg": epg, "quant_type_id": quant_type_id, "is_k_full": is_k_full,
        "input_dtype": input_dtype, "activation": activation, "slices": slices,
    }


def ncompass_moe_core(x, topk_ids, topk_weights, operands, comm, ep_rank, ep_world,
                        out):
    """G-way grouped MoE: ragged all_to_all_v dispatch -> grid-capped Marlin -> combine.

    Writes the routed MoE output into `out` (pre-zeroed [T,H] tensor, allocated by caller)
    in place; does NOT include shared-expert output.

    `ep_rank` is not needed by the ragged-dispatch routing itself (destination is derived
    from each pair's global expert id, not from this rank's identity) -- it is accepted for
    API symmetry with `ep_world` / future use, unused here.
    """
    dev = x.device
    T, H = x.shape
    epg, qtid, is_k_full, in_dtype, act, slices = (
        operands["epg"], operands["quant_type_id"], operands["is_k_full"],
        operands["input_dtype"], operands["activation"], operands["slices"])
    E_LOCAL = epg * G
    # Marlin M-tile, derived from this call's routing shape (see _select_block_size_m).
    # Host-side, constant per captured graph size.
    _bsm = (_select_block_size_m(T, topk_ids.shape[1], E_LOCAL, ep_world)
            if _TILE_AUTO else _MARLIN_TILE_DEFAULT)


    # Fix the transport capacities before anything downstream reads them (the metadata
    # builder and the transport both do). Host-side integer arithmetic, cached after the
    # first call, which lands on the eager profiling forward.
    _cfg.resolve_caps(topk_ids.shape[1], ep_world)

    tk_ids = topk_ids.to(torch.long)               # [T,K] GLOBAL expert ids
    tk_w = topk_weights.to(torch.float32)          # [T,K]

    # -------- per-group ragged metadata (recomputed each step; routing is dynamic)
    # --------
    # Histogram all-gather: 1 collective (vs 3), recv counts + expert-ids derived
    # locally.
    metas = build_group_meta_hist(tk_ids, tk_w, E_LOCAL, epg, G, ep_world, ep_rank, comm, dev,
                                  r_cap=_cfg.r_cap())

    transport = _get_transport(comm, H, x.dtype, dev, ep_world, ep_rank)
    transport.step()                                        # advance decode-step epoch once
    # CAP/capture mode: gather from x_g (an extra zero dump row) and accumulate into
    # out_ext (an extra
    # discarded dump row), so padded s_orig rows (index == T) are inert. Copied back to
    # `out` at the end.
    _cap_mode = _cfg.r_cap() is not None
    x_g = torch.cat([x, x.new_zeros(1, H)], 0) if _cap_mode else x
    out_ext = torch.zeros(T + 1, H, device=dev, dtype=out.dtype) if _cap_mode else None

    def dispatch(m, g):
        recv_x = transport.dispatch(g, m, x_g)              # device-count a2a (x_g has the dump row)
        return recv_x

    def compute(m, recv_x, g):
        # shape-based guard: transport recv_x is the full CAP buffer (>0) so Marlin always
        # runs (padding
        # rows are masked via sentinel expert ids). NCCL path recv_x is R rows -> skip
        # Marlin at R==0.
        if recv_x.shape[0] == 0:
            return torch.zeros(0, H, device=dev, dtype=x.dtype)
        w = slices[g]
        # Fuse the combine send-copy: write Marlin's 2nd-GEMM output straight into group
        # g's transport
        # combine send buffer, so combine() runs pre-staged (skips its CAP-sized
        # memcpy). Only when
        # combine goes through transport in cap mode; DBG path keeps the plain buffer for
        # its NCCL ref.
        out_buf = transport.combine_send_buf(g, m) if _cap_mode else None
        # Right-size the SiLU: in cap mode the intermediate is [R_CAP, 2N] but only the
        # leading
        # R_valid rows are real (recv rows land contiguously at [0:R_valid]). num_valid
        # is a DEVICE
        # scalar = sum(recv_counts_dev) (m["R"] is a host 0 in cap mode, unusable). The
        # masked silu
        # skips padding tiles; dense fallback for non-cap mode or non-plain-SiLU
        # activations.
        _nv = None
        _slot_cnt = None
        if _cap_mode:
            # L3 P2: valid rows are WORLD disjoint runs, not a prefix, so pass the per-
            # source
            # counts and let the kernel derive validity from the row index. Under NOAG
            # these
            # counts came from the signal value -- still no host sync, still capturable.
            _slot_cnt = m["recv_counts_dev"]

        def act_fn(activation, out_, in_, clamp_limit=None, alpha=1.0, beta=0.0):
            if (_slot_cnt is not None and activation == MoEActivation.SILU
                    and clamp_limit is None):
                from vllm.model_executor.layers.fused_moe.ncompass_transport import SLOT_STRIDE
                return slotted_silu_and_mul(out_, in_, _slot_cnt,
                                            SLOT_STRIDE(transport.p_cap), transport.world)
            if _nv is not None and activation == MoEActivation.SILU and clamp_limit is None:
                return masked_silu_and_mul(out_, in_, _nv)
            return apply_moe_activation(activation, out_, in_,
                                        clamp_limit=clamp_limit, alpha=alpha, beta=beta)

        y = fused_marlin_moe(
            hidden_states=recv_x, w1=w["w1"], w2=w["w2"], bias1=None, bias2=None,
            w1_scale=w["w1_scale"], w2_scale=w["w2_scale"],
            topk_weights=m["recv_topk_w"], topk_ids=m["recv_topk_ids"],
            quant_type_id=qtid, global_num_experts=epg, activation=act, expert_map=None,
            w1_zeros=w["w1_zp"], w2_zeros=w["w2_zp"],
            g_idx1=w["g_idx1"], g_idx2=w["g_idx2"], sort_indices1=w["si1"], sort_indices2=w["si2"],
            is_k_full=is_k_full, input_dtype=in_dtype,
            block_size_m_override=_bsm,
            activation_func=act_fn,  # device-masked SiLU in cap mode (dense fallback otherwise)
            skip_topk_reduce=True,   # recv_topk_ids is [R,1] -> topk==1, reduce is identity
            output=out_buf,          # fused: 2nd-GEMM writes into the combine send buffer
        )
        return y

    def combine(m, y, g):
        y_back = transport.combine(g, m, y)                 # reverse device-count a2a
        if out_ext is not None and _FUSED_EPI and m.get("inv_idx") is not None:
            # B2a: ONE kernel replaces upcast + multiply + downcast + index_add_ (34.45
            # us/layer at
            # the serve shape). Token-major and deterministic; see
            # fused_combine_epilogue.
            # ACC=(g>0): group 0 STORES (every token row is written, zero if it has no
            # pairs here),
            # later groups accumulate. out_ext is still zeroed upstream, so storing is
            # safe either way.
            fused_combine_epilogue(out_ext, y_back, m["inv_idx"], m["s_weight_flat"],
                                   m["T_tokens"] + 1, accumulate=(g > 0))
        elif out_ext is not None:
            # full static scatter: valid rows -> real origin; padding rows -> discarded
            # dump row T.
            n = m["s_orig"].shape[0]
            out_ext.index_add_(0, m["s_orig"], (y_back[:n].float() * m["s_weight"]).to(out.dtype))
        elif m["P"] > 0:
            out.index_add_(0, m["s_orig"], (y_back.float() * m["s_weight"]).to(out.dtype))

    # -------- pipelined dispatch/compute/combine: dispatch/combine on `main`, compute
    # on `cs`.
    cs = _cs()
    main = torch.cuda.current_stream()
    d_evt = [torch.cuda.Event() for _ in range(G)]
    recv = [None] * G
    recv[0] = dispatch(metas[0], 0)
    d_evt[0].record(main)
    for g in range(G):
        m = metas[g]
        cs.wait_event(d_evt[g])
        with torch.cuda.stream(cs):
            y = compute(m, recv[g], g)
        m_evt = torch.cuda.Event()
        m_evt.record(cs)
        if g + 1 < G:
            recv[g + 1] = dispatch(metas[g + 1], g + 1)  # overlaps compute(g) on main
            d_evt[g + 1].record(main)
        main.wait_event(m_evt)
        combine(m, y, g)

    if out_ext is not None:
        out.copy_(out_ext[:T])                            # routed output back to `out` (drop dump row T)
