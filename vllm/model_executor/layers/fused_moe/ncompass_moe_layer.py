# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ncompass: vLLM shell for the expert-group MoE core.

Thin adapter around ncompass_moe_core.ncompass_moe_core: resolves the EP process group /
rank / world size, builds+caches the per-layer Marlin weight slices via
ncompass_moe_core.build_marlin_operands, computes the shared expert, calls the layer-agnostic
core, then adds the shared-expert output. See ncompass_moe_core.py's docstring for the
core itself.

The shipped configuration is G=1 (ncompass_moe_config.EXPERT_GROUPS): a single expert group,
so the G-way pipeline degenerates to one dispatch/compute/combine. The G>1 machinery is
retained because the core is written group-generically, not because G>1 is exercised.

Correctness anchor = AgRs baseline (harness rel_l2 ~5.6e-3). Live output is garbage (dummy KV);
validate via 0-fail + math-match. Assumes LINEAR expert placement (rank r owns globals
[r*E_LOCAL,(r+1)*E_LOCAL)) -- sanity-checked + logged on first call.
"""

import os

import torch

from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import ncompass_moe_config as _cfg
from vllm.model_executor.layers.fused_moe.ncompass_moe_core import (
    G,
    ncompass_moe_core,
    build_marlin_operands,
)

logger = init_logger(__name__)


def _cached_marlin_operands(layer, moe_method):
    """Static per-group weight slices (cached on the layer; weights never change)."""
    cached = getattr(layer, "_ncompass_marlin_operands", None)
    if cached is not None:
        return cached
    mx = moe_method.moe_kernel.fused_experts     # MarlinExperts
    cached = build_marlin_operands(
        layer.w13_weight, layer.w2_weight,
        mx.w1_scale, mx.w2_scale, mx.w1_zp, mx.w2_zp,
        mx.w13_g_idx, mx.w2_g_idx, mx.w13_g_idx_sort_indices, mx.w2_g_idx_sort_indices,
        mx.quant_type_id, mx.is_k_full, getattr(mx, "input_dtype", None),
        layer.activation,
    )
    layer._ncompass_marlin_operands = cached
    return cached



def ncompass_moe_apply(layer, x, topk_weights, topk_ids, moe_method,
                      shared_experts, shared_experts_input):
    dev = x.device
    T, H = x.shape
    operands = _cached_marlin_operands(layer, moe_method)
    ep = get_ep_group()
    WORLD = ep.world_size
    comm = ep.device_group
    E_LOCAL = layer.global_num_experts // WORLD

    # first-call sanity: linear placement (expert_map[RANK*E_LOCAL + i] == i)
    #
    # SKIPPED DURING CUDA-GRAPH CAPTURE. `torch.equal` returns a Python bool, so it forces a
    # device-to-host sync -- illegal while a stream is capturing ("operation not permitted when
    # stream is capturing"). This was latent for as long as the first call into this function
    # happened during an eager warmup, which set `_ncompass_checked` before capture began. Decode-only
    # routing removed that guarantee: non-uniform warmup batches now go to the stock kernel, so the
    # FIRST entry here can be a captured one, and the one-shot check then fires inside capture.
    #
    # Skipping is correct rather than merely convenient -- this is a diagnostic, not a
    # precondition, and the flag is deliberately NOT set when skipped so the check still runs on
    # the first eager call if one ever occurs.
    _capturing = False
    try:
        _capturing = torch.cuda.is_current_stream_capturing()
    except Exception:
        pass
    if not _capturing and not getattr(layer, "_ncompass_checked", False):
        layer._ncompass_checked = True
        em = layer.expert_map
        rk = ep.rank_in_group
        ok = "n/a"
        if em is not None:
            loc = em[rk * E_LOCAL:(rk + 1) * E_LOCAL]
            ok = bool(torch.equal(loc.to(torch.long),
                                  torch.arange(E_LOCAL, device=loc.device, dtype=torch.long)))
        logger.info("ncompass MoE: WORLD=%d E_LOCAL=%d epg=%d linear_placement=%s",
                    WORLD, E_LOCAL, operands["epg"], ok)

    # ROUTED OUTPUT ONLY -- do NOT compute or add the shared expert here.
    #
    # MoERunner already owns the shared expert: `_maybe_apply_shared_experts` runs it exactly
    # once (order-gated in SharedExperts.forward), stashes it in `_output`, and moe_runner.py
    # sums it with our return value at `result = shared_output + fused_output`. Computing it
    # here as well ran the gate_up/silu/down MLP TWICE per layer (trace: those GEMMs at 2.0/layer
    # against stock's 1.0, ~45 us/layer) and, because `_maybe_apply_routed_scale_to_output`
    # scales fused_output by routed_scaling_factor, made the total
    # `shared*(1+routed_scaling_factor) + routed*routed_scaling_factor` instead of
    # `shared + routed*routed_scaling_factor`.
    #
    # Calling `shared_experts._layer()` directly also bypasses the order gate, which is why the
    # runner's own duplicate-guard assert never fired. `shared_experts` / `shared_experts_input`
    # stay in the signature because the modular-kernel bridge passes them positionally.
    out = torch.zeros(T, H, device=dev, dtype=x.dtype)
    ncompass_moe_core(x, topk_ids, topk_weights, operands, comm, ep.rank_in_group, WORLD, out)
    return out
