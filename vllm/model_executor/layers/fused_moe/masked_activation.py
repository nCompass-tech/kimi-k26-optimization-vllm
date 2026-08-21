# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-masked gated SiLU activation for the ragged G2 MoE pipeline.

The G2 pipeline runs the expert GEMMs on a static CAP-sized recv buffer
([R_CAP, .]) but only the leading `num_valid` rows (a DEVICE scalar) carry real
tokens; the rest are sentinel padding. The stock `silu_and_mul` is a dense
elementwise pass over all R_CAP rows -> ~94% wasted work at decode fill.

`masked_silu_and_mul` early-exits fully-padding row-tiles via a device-loaded
`num_valid`, so the work is proportional to the valid tokens. The grid is static
(sized to R_CAP), so padding tiles cost only a predicated return -> CUDA-graph
capturable, no host sync. Adapts vLLM `fused_moe_kernel`'s
`num_tokens_post_padded` tile-skip idiom.
"""


import torch
import triton
import triton.language as tl


@triton.jit
def _masked_silu_mul_kernel(
    out_ptr, in_ptr, nv_ptr, M, N,
    s_in_m, s_in_n, s_out_m, s_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    nv = tl.load(nv_ptr)                       # device: # valid leading rows
    row0 = pid_m * BLOCK_M
    if row0 >= nv:                             # whole tile is padding -> skip
        return
    rows = row0 + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < nv                             # valid rows within this tile
    cm = cols < N
    mask = rm[:, None] & cm[None, :]
    gate_p = in_ptr + rows[:, None] * s_in_m + cols[None, :] * s_in_n
    up_p = in_ptr + rows[:, None] * s_in_m + (N + cols[None, :]) * s_in_n
    gate = tl.load(gate_p, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_p, mask=mask, other=0.0).to(tl.float32)
    res = (gate * tl.sigmoid(gate) * up).to(out_ptr.dtype.element_ty)
    o_p = out_ptr + rows[:, None] * s_out_m + cols[None, :] * s_out_n
    tl.store(o_p, res, mask=mask)


@triton.jit
def _slotted_silu_mul_kernel(
    out_ptr, in_ptr, cnt_ptr, M, N,
    s_in_m, s_in_n, s_out_m, s_out_n,
    WORLD: tl.constexpr, STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Slot-aware twin of `_masked_silu_mul_kernel` (L3 P2).

    The dense-prefix kernel assumes valid rows are [0, nv). Under the slotted layout they are
    WORLD disjoint runs -- source s owns [s*STRIDE, s*STRIDE + cnt_s) -- so a prefix length cannot
    describe them, and using one computes GAP rows while leaving every real row untouched (audit B1:
    silent NaN, no shape error).

    INDEX-ONLY: no rows are moved. Validity is derived arithmetically from the row index
    (s = r // STRIDE, o = r % STRIDE, valid iff o < cnt[s]), so compaction costs one small gather of
    WORLD counts per tile instead of a copy of the whole [R_CAP, 2N] intermediate."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    slot = rows // STRIDE
    off = rows % STRIDE
    c = tl.load(cnt_ptr + slot, mask=slot < WORLD, other=0)
    rm = (off < c) & (rows < M)
    # Uniform across the block (a reduction), so this is a scalar branch, not a divergent one, and
    # it carries no collective -- safe to early-return. Most tiles are padding at decode fill.
    if tl.sum(rm.to(tl.int32), axis=0) == 0:
        return
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = rm[:, None] & (cols < N)[None, :]
    gate_p = in_ptr + rows[:, None] * s_in_m + cols[None, :] * s_in_n
    up_p = in_ptr + rows[:, None] * s_in_m + (N + cols[None, :]) * s_in_n
    gate = tl.load(gate_p, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_p, mask=mask, other=0.0).to(tl.float32)
    res = (gate * tl.sigmoid(gate) * up).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + rows[:, None] * s_out_m + cols[None, :] * s_out_n, res, mask=mask)


def slotted_silu_and_mul(out: torch.Tensor, inp: torch.Tensor,
                         cnt: torch.Tensor, stride: int, world: int) -> torch.Tensor:
    """out[r] = silu(inp[r,:N]) * inp[r,N:] for rows inside a slot's valid run; others untouched.

    cnt: device int64 [WORLD] per-source row counts (under NOAG these come from the signal decode).
    Static grid over M -> capturable, no host sync."""
    M, twoN = inp.shape
    N = out.shape[1]
    assert twoN == 2 * N, f"expected [M,2N] gated input, got {inp.shape} vs out N={N}"
    assert M >= world * stride, (
        f"slotted silu needs M >= WORLD*STRIDE = {world*stride}, got {M} -- rows beyond the last "
        "slot would be read with an out-of-range slot index")
    # Block shape settled by sweep. The static grid is sized to R_CAP, but at decode fill only
    # ~6% of rows are valid, so MOST CTAs only early-return -- measured 8.6 us/call of which
    # ~5.8 us is that fixed grid cost, not work. Fewer/larger blocks trade intra-tile waste
    # against CTA count; 32x512 is the measured winner and is what the shipped arm runs.
    BLOCK_M = 32
    BLOCK_N = 512
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _slotted_silu_mul_kernel[grid](
        out, inp, cnt.to(torch.int32), M, N,
        inp.stride(0), inp.stride(1), out.stride(0), out.stride(1),
        WORLD=world, STRIDE=stride, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


def masked_silu_and_mul(out: torch.Tensor, inp: torch.Tensor,
                        num_valid: torch.Tensor) -> torch.Tensor:
    """out[:nv] = silu(inp[:nv, :N]) * inp[:nv, N:2N]; rows >= nv left untouched.

    out:  [M, N]   inp: [M, 2N]   num_valid: device int scalar (0-dim or [1]).
    Only the plain gated-SiLU case (Kimi); callers route clamp/other acts to the
    dense op. Static grid over M -> capturable.
    """
    M, twoN = inp.shape
    N = out.shape[1]
    assert twoN == 2 * N, f"expected [M,2N] gated input, got {inp.shape} vs out N={N}"
    nv = num_valid.reshape(()).to(torch.int32)
    BLOCK_M, BLOCK_N = 32, 512
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _masked_silu_mul_kernel[grid](
        out, inp, nv, M, N,
        inp.stride(0), inp.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


if __name__ == "__main__":
    # Single-GPU sanity vs a pure-torch reference (no vllm custom-ops dependency).
    torch.manual_seed(0)
    dev = "cuda"
    for M, N, nv_val in [(11520, 2048, 640), (256, 1024, 256), (100, 512, 0), (128, 256, 5)]:
        inp = torch.randn(M, 2 * N, device=dev, dtype=torch.bfloat16)
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        out.fill_(float("nan"))                # so untouched rows are detectable
        nv = torch.tensor(nv_val, device=dev, dtype=torch.int32)
        masked_silu_and_mul(out, inp, nv)
        g = inp[:nv_val, :N].float()
        ref = (g * torch.sigmoid(g) * inp[:nv_val, N:].float()).to(torch.bfloat16)
        got = out[:nv_val]
        rel = (got.float() - ref.float()).norm() / (ref.float().norm() + 1e-9)
        untouched_ok = bool(torch.isnan(out[nv_val:]).all()) if nv_val < M else True
        print(f"M={M} N={N} nv={nv_val}: rel_l2={rel:.2e} "
              f"untouched_rows_preserved={untouched_ok} "
              f"{'PASS' if (rel < 1e-2 and untouched_ok) else 'FAIL'}")


@triton.jit
def _k_fused_combine_epilogue(
    out_ptr, y_ptr, inv_ptr, w_ptr, T, H, MAXP,
    s_out_m, s_y_m,
    K: tl.constexpr, BLOCK_H: tl.constexpr, ACC: tl.constexpr,
):
    """B2a: the whole MoE combine epilogue in ONE kernel.

    Replaces `out.index_add_(0, s_orig, (y_back.float() * s_weight).to(dtype))`, which PyTorch
    expands into FOUR kernels (upcast, multiply, downcast, atomic scatter) totalling 34.45 us/layer
    at the serve shape. Three of those materialise a full [P_CAP, H] intermediate in HBM.

    This reduces TOKEN-MAJOR instead of pair-major: each program owns one token and walks that
    token's <= K contributing rows via the inverse index, accumulating in fp32 registers and storing
    bf16 once. That removes the intermediates AND the atomics -- the result is deterministic, which
    `index_add_`'s atomic accumulation is not.

    `inv[t, k]` holds the row position of the token's k-th pair IN THIS GROUP, or -1 when the token
    has fewer than k pairs here. A token can legitimately have ZERO pairs in a group (its top-K
    experts may all live in other groups), in which case every slot is -1 and the token contributes
    zero -- so ACC=0 must still STORE, not skip, or that row would be left uninitialised.

    Donor: the top-k scatter/reduce benchmark in the nCompass KB measured a static Triton reducer at
    0.1274 ms vs index_add_ 0.6596 ms at TOPK=8 (5.2x), with atomics and reorder+sum both worse.
    """
    t = tl.program_id(0)
    hb = tl.program_id(1)
    cols = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    cm = cols < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(K):
        p = tl.load(inv_ptr + t * K + k)
        valid = p >= 0
        # clamp before use: a masked load does not dereference, but the ADDRESS arithmetic still
        # runs, and p = -1 would compute a wildly out-of-range offset.
        pc = tl.where(valid, p, 0)
        w = tl.load(w_ptr + pc, mask=valid, other=0.0)
        v = tl.load(y_ptr + pc * s_y_m + cols, mask=cm & valid, other=0.0).to(tl.float32)
        acc += v * w
    o = out_ptr + t * s_out_m + cols
    if ACC:
        acc += tl.load(o, mask=cm, other=0.0).to(tl.float32)
    tl.store(o, acc.to(out_ptr.dtype.element_ty), mask=cm)


def fused_combine_epilogue(out, y_back, inv_idx, s_weight, n_tokens, accumulate):
    """out[t] (+)= sum_k y_back[inv[t,k]] * s_weight[inv[t,k]], skipping inv == -1.

    out: [T+1, H] or [T, H]   y_back: [MAXP, H]   inv_idx: [T+1, K] int32   s_weight: [MAXP] fp32
    Static grid over tokens x H-blocks -> CUDA-graph capturable, no host sync.
    """
    H = out.shape[1]
    K = inv_idx.shape[1]
    BLOCK_H = 1024
    grid = (n_tokens, triton.cdiv(H, BLOCK_H))
    _k_fused_combine_epilogue[grid](
        out, y_back, inv_idx, s_weight, n_tokens, H, y_back.shape[0],
        out.stride(0), y_back.stride(0),
        K=K, BLOCK_H=BLOCK_H, ACC=(1 if accumulate else 0), num_warps=8,
    )
    return out
