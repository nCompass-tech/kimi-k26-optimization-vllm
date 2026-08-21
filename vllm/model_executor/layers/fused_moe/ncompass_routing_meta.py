# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ncompass: fused on-device routing metadata for the G-way grouped MoE pipeline.

Replaces the per-op metadata storm in ncompass_moe_core (`(%//)` -> `nonzero` -> `argsort`
-> `bincount` -> gathers, ~40 tiny launches/layer + ~16 D2H scalar round-trips + ~10 stream
syncs) with two Triton kernels doing a counting-sort-by-destination:

  1. _count_kernel   : for each (token,slot) pair, compute its expert-group and owner rank,
                       atomic-increment send_counts[group, dest].
  2. _scatter_kernel : re-scan the pairs, atomic-bump a per-bucket cursor, and scatter
                       (token_row, weight, owner-local-expert) into the dest-ordered slot.

Structure adapted from NVIDIA Megatron-LM's `_scatter_token_indices_kernel` /
`_moe_align_block_size_cuda_graphable` (megatron/core/inference/moe/vllm_fused_moe.py):
same atomic-counter counting-sort, bucketed by owner RANK here instead of local expert.

Within-bucket order is arbitrary (atomic race) but irrelevant to correctness: dispatch sends
each dest's whole chunk, compute carries the per-row expert id, and combine scatters back via
`s_orig` with an atomic `index_add_`. The AgRs oracle (harness rel_l2 gate) is the anchor.

`torch.distributed.all_to_all_single` still needs HOST split-size lists, so one irreducible D2H
of the [2,G,WORLD] count block remains (vs the old ~16 scalar copies). Counts are computed on
device; the count-exchange itself uses equal (static) splits so it needs no host lists.
"""


import torch
import torch.distributed as dist
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe import ncompass_moe_config as _cfg

# B5/B6 lever L1: when the recv buffer is slot-addressed, the row->expert mapping must
# follow the slots rather than dense source order (see build_recv_ids_slotted).
_SLOTTED = _cfg.SLOTTED
# L3 P1b-2: with slots, every recv-side quantity is recoverable from the transport itself
# (counts from the signal value, rev_recv_base + per-expert counts from the header row),
# so the histogram AllGather has no remaining consumer and is skipped entirely.
_NOAG = _cfg.NOAG
# P3 scratch hoisting; =0 restores the per-step allocate+fill path for bisecting.
# B2a: fused combine epilogue (one Triton kernel replacing upcast+mul+downcast+index_add_).
_FUSED_EPI = _cfg.FUSED_EPI
# B2b: skip metadata work the transport already supplies, and fuse adjacent kernels.
_B2B = _cfg.B2B


@triton.jit
def _count_kernel(tk_ids_ptr, send_counts_ptr, n_pairs, E_LOCAL, epg, WORLD,
                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    e = tl.load(tk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    grp = (e % E_LOCAL) // epg          # [0, G)
    dest = e // E_LOCAL                 # [0, WORLD)  (LINEAR placement)
    tl.atomic_add(send_counts_ptr + grp * WORLD + dest, 1, mask=mask)


@triton.jit
def _scatter_kernel(tk_ids_ptr, tk_w_ptr, base_ptr, counters_ptr,
                    s_orig_ptr, s_weight_ptr, s_expert_ptr,
                    n_pairs, K, E_LOCAL, epg, WORLD, MAXP, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    e = tl.load(tk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    w = tl.load(tk_w_ptr + offs, mask=mask, other=0.0)
    grp = (e % E_LOCAL) // epg
    dest = e // E_LOCAL
    expert_local = e % E_LOCAL           # owner's local slot [0, E_LOCAL)
    trow = offs // K                     # token row = flat pair idx // topk
    gd = grp * WORLD + dest
    pos = tl.load(base_ptr + gd, mask=mask, other=0) + tl.atomic_add(
        counters_ptr + gd, 1, mask=mask)
    out = grp * MAXP + pos               # dest-ordered slot in group g's buffer
    tl.store(s_orig_ptr + out, trow.to(tl.int64), mask=mask)
    tl.store(s_weight_ptr + out, w, mask=mask)
    tl.store(s_expert_ptr + out, expert_local.to(tl.int64), mask=mask)


# ---------------------------------------------------------------------------
# Fully-fused device-side metadata (cap-mode). Two Triton kernels replace the
# ~30 tiny torch scan/searchsorted ops of build_group_meta_hist with a
# shared-histogram + block-scan + grid-strided-materialize pattern (adapted from
# FlashInfer routingIndicesOffsetsKernel / vLLM moe_align_block_size_kernel).
# Everything is device-resident (NO D2H) and static-shape (capturable).
# ---------------------------------------------------------------------------


@triton.jit
def _meta_offsets_kernel(
    my_hist_ptr, H_ptr,
    send_counts_ptr, recv_counts_ptr, send_start_ptr, recv_base_ptr,
    rev_recv_base_ptr, base_ptr,
    ep_rank, WORLD: tl.constexpr, epg: tl.constexpr, EPG_P: tl.constexpr, RECV: tl.constexpr,
    E_LOCAL: tl.constexpr, NE: tl.constexpr, NB: tl.constexpr,
):
    """One program per group g. From the local histogram mh and the gathered H,
    emit every per-group offset tensor in a single launch:
      send_counts[g,d], recv_counts[g,s], send_start[g,d] (excl-scan over dest),
      recv_base[g,d]  = sum_{src<ep_rank} std[src,d],
      rev_recv_base[g,s] = sum_{dest<ep_rank} std[s,dest],
      base[g, d*epg+eig] = excl-scan over the (dest,eig) buckets of mh.
    Matches build_group_meta_hist:133-157 exactly. epg is padded to EPG_P (pow2) + masked
    (tl.arange needs pow2); padding entries carry 0 counts so scans are unaffected."""
    g = tl.program_id(0)
    d = tl.arange(0, WORLD)                                # WORLD is pow2
    ep = tl.arange(0, EPG_P)
    realm = ep < epg                                       # [EPG_P]
    # --- send side: mh_g[d, eig] = my_hist[d*E_LOCAL + g*epg + eig], padding cols = 0 ---
    mh_cols = d[:, None] * E_LOCAL + g * epg + ep[None, :]         # [W, EPG_P]
    mh = tl.load(my_hist_ptr + mh_cols, mask=realm[None, :], other=0).to(tl.int32)
    send_counts = tl.sum(mh, axis=1)                              # [W]
    send_start = tl.cumsum(send_counts, axis=0) - send_counts     # excl [W]
    flat = tl.reshape(mh, [WORLD * EPG_P])                        # padded, d-major eig-minor
    cum = tl.cumsum(flat, axis=0) - flat                         # exclusive over padded order
    cum2 = tl.reshape(cum, [WORLD, EPG_P])
    base_idx = d[:, None] * epg + ep[None, :]                     # COMPACT [W, EPG_P]
    tl.store(base_ptr + g * NB + base_idx, cum2.to(tl.int32), mask=realm[None, :])
    tl.store(send_counts_ptr + g * WORLD + d, send_counts.to(tl.int64))
    tl.store(send_start_ptr + g * WORLD + d, send_start.to(tl.int64))
    # --- recv side. B2b: SKIPPED under NOAG, where all three outputs are provably dead:
    #   recv_counts   -> overwritten in NcompassMoETransport.dispatch by b["dcnt"] (decoded from the signal value)
    #   rev_recv_base -> overwritten by b["hdr_rrb"] (read from the header rows)
    #   recv_base     -> dead outright; the slot layout makes it myrank*STRIDE, computable locally
    # It is a `static_range(epg)` loop (24 iterations at epg=24), each loading a [WORLD, WORLD] tile
    # of H, and it dominated this kernel's 4.81 us/layer. H itself is all-zero under NOAG, so the
    # loop was summing zeros to produce values that are then discarded.
    if RECV:
        s = tl.arange(0, WORLD)
        std_g = tl.zeros([WORLD, WORLD], dtype=tl.int64)
        for e in tl.static_range(epg):
            cols_de = d * E_LOCAL + g * epg + e                  # [W] (dest)
            H_e = tl.load(H_ptr + s[:, None] * NE + cols_de[None, :]).to(tl.int64)
            std_g += H_e                                         # [src, dest]
        dmask = (d == ep_rank).to(tl.int64)
        slt = (s < ep_rank).to(tl.int64)
        recv_counts = tl.sum(std_g * dmask[None, :], axis=1)     # [W] std_g[:, ep_rank]
        recv_base = tl.sum(std_g * slt[:, None], axis=0)         # [W] over src<ep_rank
        rev_recv_base = tl.sum(std_g * slt[None, :], axis=1)     # [W] over dest<ep_rank
        tl.store(recv_counts_ptr + g * WORLD + d, recv_counts)
        tl.store(recv_base_ptr + g * WORLD + d, recv_base)
        tl.store(rev_recv_base_ptr + g * WORLD + d, rev_recv_base)


@triton.jit
def _recv_ids_kernel(
    H_ptr, recv_ids_ptr, ep_rank, r_cap,
    WORLD: tl.constexpr, epg: tl.constexpr, EPG_P: tl.constexpr, E_LOCAL: tl.constexpr,
    NE: tl.constexpr, BLOCK: tl.constexpr,
):
    """grid (G, ceil(r_cap/BLOCK)). recv_ids[g,p] = group-relative expert id of recv slot p, or
    sentinel epg for p>=R. hsub[src*epg+eig]=H[src, ep_rank*E_LOCAL+g*epg+eig]; offs=cumsum(hsub);
    bucket = searchsorted(offs,p,right)=count(offs<=p); eig=bucket%epg. Matches :184-188. epg padded
    to EPG_P (pow2); padding offs set to r_cap+1 (never <= p) so they don't inflate the count."""
    g = tl.program_id(0)
    pid_p = tl.program_id(1)
    s = tl.arange(0, WORLD)
    ep = tl.arange(0, EPG_P)
    realm = ep < epg
    cols = s[:, None] * NE + ep_rank * E_LOCAL + g * epg + ep[None, :]   # [W, EPG_P]
    hsub = tl.load(H_ptr + cols, mask=realm[None, :], other=0).to(tl.int32)
    flat = tl.reshape(hsub, [WORLD * EPG_P])                     # padded, src-major eig-minor
    cum = tl.cumsum(flat, axis=0)                               # inclusive over padded order
    cum2 = tl.reshape(cum, [WORLD, EPG_P])
    offs2 = tl.where(realm[None, :], cum2, r_cap + 1)           # padding -> never <= p
    offs = tl.reshape(offs2, [WORLD * EPG_P])                    # [W*EPG_P]
    pos = pid_p * BLOCK + tl.arange(0, BLOCK)                    # [BLOCK]
    pmask = pos < r_cap
    le = (offs[None, :] <= pos[:, None]).to(tl.int32)           # [BLOCK, W*EPG_P]
    bucket = tl.sum(le, axis=1)                                  # compact searchsorted right
    eig_out = tl.where(bucket < WORLD * epg, bucket % epg, epg)
    tl.store(recv_ids_ptr + g * r_cap + pos, eig_out.to(tl.int64), mask=pmask)


@triton.jit
def _recv_ids_slotted_kernel(
    H_ptr, recv_ids_ptr, ep_rank, r_cap,
    WORLD: tl.constexpr, epg: tl.constexpr, EPG_P: tl.constexpr, E_LOCAL: tl.constexpr,
    NE: tl.constexpr, P_CAP: tl.constexpr, PCAP_P: tl.constexpr,
):
    """SLOTTED twin of `_recv_ids_kernel` (B5/B6 lever L1).

    The dense kernel encodes "row p is the p-th row in DENSE source order" via a global cumsum. Under
    the slotted layout row p means (slot p//P_CAP, offset p%P_CAP) and source s's rows are
    RIGHT-ALIGNED inside its slot, so that mapping is wrong for every row -- which is why the layout
    change cannot be validated without this.

    grid (G, WORLD): one program per (group, source), writing that source's slot.
      cnt_s   = sum_eig hsub[s, eig]          rows source s sends me in this group
      start   = P_CAP - 1 - cnt_s             right-aligned; P_CAP-1 is the reserved header row
      row j of source s -> eig = count(cumsum(hsub[s,:]) <= j)   (same per-source order as dense)
      everything outside [start, P_CAP-1) is padding -> sentinel `epg`, which Marlin already masks.
    """
    g = tl.program_id(0)
    s = tl.program_id(1)
    ep = tl.arange(0, EPG_P)
    realm = ep < epg
    cols = s * NE + ep_rank * E_LOCAL + g * epg + ep
    hsub = tl.load(H_ptr + cols, mask=realm, other=0).to(tl.int32)     # [EPG_P]
    offs = tl.cumsum(hsub, axis=0)                                     # inclusive, per-source
    cnt_s = tl.sum(hsub, axis=0)
    offs = tl.where(realm, offs, r_cap + 1)                            # padding never <= j
    # LEFT-ALIGNED slots (L3): source s owns [s*STRIDE, s*STRIDE + P_CAP], data at offsets
    # [0, cnt_s), header at cnt_s, padding after. STRIDE = P_CAP+1 -- see SLOT_STRIDE in
    # ncompass_transport.py; the two MUST agree or rows land in the wrong slot.
    # (The previous right-aligned window existed so a put always ENDED at a fixed address for the
    # retired sentinel poll; put+signal removed that requirement.)
    STRIDE = P_CAP + 1
    pos = tl.arange(0, PCAP_P)                                         # offset within the slot
    valid = pos < cnt_s                                                # data rows only
    le = (offs[None, :] <= pos[:, None]).to(tl.int32)                  # [PCAP_P, EPG_P]
    bucket = tl.sum(le, axis=1)
    eig_out = tl.where(valid & (bucket < epg), bucket, epg)            # header+padding -> epg
    row = s * STRIDE + pos
    tl.store(recv_ids_ptr + g * r_cap + row, eig_out.to(tl.int64), mask=(pos < STRIDE))


def build_recv_ids_slotted(H, ep_rank, r_cap, p_cap, WORLD, epg, E_LOCAL, G, dev):
    """Slot-aware recv expert ids. Returns [G, r_cap], same contract as build_recv_ids_fused."""
    NE = WORLD * E_LOCAL
    EPG_P = triton.next_power_of_2(epg)
    PCAP_P = triton.next_power_of_2(p_cap)
    assert r_cap >= WORLD * (p_cap + 1), (
        f"slotted recv needs WORLD*(P_CAP+1)={WORLD*(p_cap+1)} rows, got r_cap={r_cap} "
        "(P0 audit F2: the header row makes usable slot capacity P_CAP, so the stride is P_CAP+1)")
    # full sentinel prefill: the kernel writes WORLD*(P_CAP+1) rows; any tail beyond that must still
    # be padding, not uninitialised int64 (P0 audit V3).
    recv_ids = torch.full((G, r_cap), epg, dtype=torch.int64, device=dev)
    _recv_ids_slotted_kernel[(G, WORLD)](
        H, recv_ids, ep_rank, r_cap,
        WORLD=WORLD, epg=epg, EPG_P=EPG_P, E_LOCAL=E_LOCAL, NE=NE,
        P_CAP=p_cap, PCAP_P=PCAP_P,
    )
    return recv_ids


def build_offsets_fused(my_hist, H, ep_rank, WORLD, epg, E_LOCAL, G, dev, recv=True):
    """Fused replacement for the scan block (send/recv counts, bases, offsets).

    recv=False (B2b, NOAG only): skip the recv half entirely. Its three outputs are all overwritten
    or dead once the transport supplies them -- see the RECV comment in `_meta_offsets_kernel`. The
    returned recv tensors are then UNINITIALISED and must not be read; NcompassMoETransport.dispatch replaces the
    two that matter before any consumer sees them."""
    NE = WORLD * E_LOCAL
    NB = WORLD * epg
    EPG_P = triton.next_power_of_2(epg)
    send_counts = torch.empty(G, WORLD, dtype=torch.int64, device=dev)
    recv_counts = torch.empty(G, WORLD, dtype=torch.int64, device=dev)
    send_start = torch.empty(G, WORLD, dtype=torch.int64, device=dev)
    recv_base = torch.empty(G, WORLD, dtype=torch.int64, device=dev)
    rev_recv_base = torch.empty(G, WORLD, dtype=torch.int64, device=dev)
    base = torch.empty(G, NB, dtype=torch.int32, device=dev)
    _meta_offsets_kernel[(G,)](
        my_hist, H, send_counts, recv_counts, send_start, recv_base, rev_recv_base, base,
        ep_rank, WORLD=WORLD, epg=epg, EPG_P=EPG_P, RECV=(1 if recv else 0),
        E_LOCAL=E_LOCAL, NE=NE, NB=NB,
    )
    return send_counts, recv_counts, send_start, recv_base, rev_recv_base, base


def build_recv_ids_fused(H, ep_rank, r_cap, WORLD, epg, E_LOCAL, G, dev):
    """Fused replacement for the per-group searchsorted recv_topk_ids build. Returns [G, r_cap]."""
    NE = WORLD * E_LOCAL
    EPG_P = triton.next_power_of_2(epg)
    recv_ids = torch.empty(G, r_cap, dtype=torch.int64, device=dev)
    BLOCK = 256
    grid = (G, triton.cdiv(r_cap, BLOCK))
    _recv_ids_kernel[grid](
        H, recv_ids, ep_rank, r_cap,
        WORLD=WORLD, epg=epg, EPG_P=EPG_P, E_LOCAL=E_LOCAL, NE=NE, BLOCK=BLOCK,
    )
    return recv_ids


@triton.jit
def _hist_kernel(tk_ids_ptr, hist_ptr, n_pairs, BLOCK: tl.constexpr):
    """my_hist[global_expert] += 1 per (token,slot) pair. Fixed-size output (NE), no host sync
    (unlike torch.bincount, which D2H-syncs to size its output)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    e = tl.load(tk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    tl.atomic_add(hist_ptr + e, 1, mask=mask)


@triton.jit
def _scatter_kernel_hist(tk_ids_ptr, tk_w_ptr, base_ptr, counters_ptr,
                         s_orig_ptr, s_weight_ptr, inv_ptr, tok_ctr_ptr, TP1,
                         n_pairs, K, E_LOCAL, epg, WORLD, MAXP, NB, BLOCK: tl.constexpr,
                         INV: tl.constexpr):
    """Counting-sort into per-group buffers keyed by (dest, expert-within-group):
    bucket = dest*epg + eig, NB=WORLD*epg buckets/group. Stores s_orig + s_weight only
    (recv-expert-ids are derived from the gathered histogram, so no s_expert exchange)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    e = tl.load(tk_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    w = tl.load(tk_w_ptr + offs, mask=mask, other=0.0)
    local = e % E_LOCAL
    g = local // epg
    eig = local % epg
    dest = e // E_LOCAL
    gb = g * NB + dest * epg + eig
    trow = offs // K
    pos = tl.load(base_ptr + gb, mask=mask, other=0) + tl.atomic_add(
        counters_ptr + gb, 1, mask=mask)
    out = g * MAXP + pos
    tl.store(s_orig_ptr + out, trow.to(tl.int64), mask=mask)
    tl.store(s_weight_ptr + out, w, mask=mask)
    if INV:
        # B2a: also record the INVERSE mapping token -> its row positions in this group, so the
        # combine epilogue can reduce token-major instead of scattering pair-major with atomics.
        # `slot` is this pair's index among the token's pairs IN THIS GROUP; a token has at most K
        # of them (its top-K experts), and may have ZERO if none of its experts live in this group.
        # Rides the SAME atomic traversal that already computes `pos`, so it adds one atomic and one
        # store to a pass we were running anyway.
        slot = tl.atomic_add(tok_ctr_ptr + g * TP1 + trow, 1, mask=mask)
        tl.store(inv_ptr + (g * TP1 + trow) * K + slot, pos.to(tl.int32),
                 mask=mask & (slot < K))


_ONES_CACHE: dict = {}


def _ones_col(r_cap, dev):
    """Hoisted [r_cap,1] ones (recv_topk_w) — allocate-once, reused every step (static addr =>
    capturable), instead of a per-group per-step torch.ones launch."""
    key = (r_cap, str(dev))
    t = _ONES_CACHE.get(key)
    if t is None:
        t = torch.ones(r_cap, 1, device=dev, dtype=torch.float32)
        _ONES_CACHE[key] = t
    return t


@triton.jit
def _k_meta_reset(my_hist_ptr, counters_ptr, s_orig_ptr, s_weight_ptr,
                  tok_ctr_ptr, inv_ptr, DUMP, NE, NCNT, NSO, NTC, NINV,
                  BLOCK: tl.constexpr, INV: tl.constexpr):
    """P3: one kernel replacing the four per-step allocate+fill launches that reset the metadata
    scratch (my_hist zeros, counters zeros, s_orig := dump row T, s_weight zeros).

    These buffers have STATIC shapes under a captured graph, so re-allocating and re-filling them
    every step cost 4 separate ~1 us graph nodes for what is a single pass of trivial stores. The
    buffers are now hoisted into _SCRATCH and only reset here.

    my_hist and counters MUST be zeroed before their kernels (both accumulate with atomics);
    s_orig defaults to the dump row so pairs a group never fills stay inert.

    NE/NCNT/NSO/DUMP are RUNTIME args, deliberately NOT tl.constexpr. NSO = G*MAXP scales with the
    batch, and serve captures 42 CUDA-graph sizes -- as constexpr they would force 42 separate JIT
    compilations of this kernel on a cold cache, which is minutes of startup and a plausible cause of
    a health-gate timeout. One specialisation serves every graph size."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(my_hist_ptr + off, tl.zeros([BLOCK], tl.int32), mask=off < NE)
    tl.store(counters_ptr + off, tl.zeros([BLOCK], tl.int32), mask=off < NCNT)
    tl.store(s_orig_ptr + off, DUMP + tl.zeros([BLOCK], tl.int64), mask=off < NSO)
    tl.store(s_weight_ptr + off, tl.zeros([BLOCK], tl.float32), mask=off < NSO)
    if INV:
        # tok_ctr must be zero before the scatter's atomics; inv defaults to -1 so unfilled slots
        # are inert in the epilogue (a token with no experts in this group has ALL slots unfilled).
        tl.store(tok_ctr_ptr + off, tl.zeros([BLOCK], tl.int32), mask=off < NTC)
        tl.store(inv_ptr + off, tl.full([BLOCK], -1, tl.int32), mask=off < NINV)


_SCRATCH = {}
_HBUF = {}
_PLACEHOLDER = {}


def _h_buf(WORLD, NE, dev, zero_once):
    """Allocate-once AllGather target. When the gather runs it overwrites H entirely, so reuse is
    safe; when it does NOT run (NOAG) the buffer stays zeroed from creation and the per-step
    torch.zeros launch disappears."""
    key = (WORLD, NE, str(dev))
    t = _HBUF.get(key)
    if t is None:
        t = (torch.zeros if zero_once else torch.empty)(WORLD, NE, dtype=torch.int32, device=dev)
        _HBUF[key] = t
    return t


def _placeholder_ids(G, r_cap, epg, dev):
    """Allocate-once sentinel-filled recv-ids placeholder for the NOAG path. NcompassMoETransport.dispatch replaces
    m["recv_topk_ids"] wholesale with the header-derived ids, so this is never read -- filling it
    once (not per step) removes another fill node."""
    key = (G, r_cap, epg, str(dev))
    t = _PLACEHOLDER.get(key)
    if t is None:
        t = torch.full((G, r_cap), epg, dtype=torch.int64, device=dev)
        _PLACEHOLDER[key] = t
    return t


def _meta_scratch(NE, NCNT, NSO, dump, dev, ntc=0, ninv=0):
    """Allocate-once metadata scratch. Static addresses => capturable, and it removes the per-step
    allocator traffic as well as the fill launches. Keyed on shape so a new graph size gets its own
    set rather than silently reusing an undersized buffer."""
    key = (NE, NCNT, NSO, ntc, ninv, str(dev))
    b = _SCRATCH.get(key)
    if b is None:
        b = {
            "my_hist": torch.empty(NE, dtype=torch.int32, device=dev),
            "counters": torch.empty(NCNT, dtype=torch.int32, device=dev),
            "s_orig": torch.empty(NSO, dtype=torch.int64, device=dev),
            "s_weight": torch.empty(NSO, dtype=torch.float32, device=dev),
            "tok_ctr": torch.empty(max(ntc, 1), dtype=torch.int32, device=dev),
            "inv_idx": torch.empty(max(ninv, 1), dtype=torch.int32, device=dev),
        }
        _SCRATCH[key] = b
    n = max(NE, NCNT, NSO, ntc, ninv)
    BLOCK = 1024
    _k_meta_reset[(triton.cdiv(n, BLOCK),)](
        b["my_hist"], b["counters"], b["s_orig"], b["s_weight"],
        b["tok_ctr"], b["inv_idx"], dump,
        NE, NCNT, NSO, ntc, ninv, BLOCK=BLOCK, INV=(1 if ninv > 0 else 0))
    return b


def build_group_meta_hist(tk_ids, tk_w, E_LOCAL, epg, G, WORLD, ep_rank, comm, dev, r_cap=None):
    """Histogram-driven metadata: ONE all_gather of the per-expert histogram replaces the count-
    exchange + both recv-expert-id exchanges (3 collectives -> 1). recv_counts AND per-row recv-
    expert-ids are derived locally from the gathered matrix H. Returns the same `metas` list the
    pipeline consumes. 

    Global expert e = dest*E_LOCAL + group*epg + eig, E_LOCAL = G*epg, so my_hist.view(WORLD,G,epg)
    indexes [dest, group, eig] exactly. Only ONE D2H (the compact [2,G,WORLD] split counts).
    """
    T, K = tk_ids.shape
    n_pairs = T * K
    MAXP = n_pairs
    NE = WORLD * E_LOCAL          # global experts (384)
    NB = WORLD * epg              # buckets per group (192)
    tkf = tk_ids.reshape(-1).contiguous()
    twf = tk_w.reshape(-1).to(torch.float32).contiguous()
    grid = (triton.cdiv(n_pairs, 256),)

    # 1. per-expert histogram (Triton atomic; no host sync).
    # P3: my_hist/counters/s_orig/s_weight are hoisted into allocate-once scratch and reset by ONE
    # kernel here, replacing four per-step allocate+fill graph nodes (~4 us/layer). Safe to reset all
    # four now: nothing between here and _scatter_kernel_hist touches counters/s_orig/s_weight.
    # B2a: inverse index sizes. TP1 = T+1 (the dump row is a real token slot). K = routing top-k,
    # the MAXIMUM pairs one token can have in one group.
    _TP1 = (MAXP // K) + 1
    _ntc = G * _TP1 if _FUSED_EPI else 0
    _ninv = G * _TP1 * K if _FUSED_EPI else 0
    _scr = _meta_scratch(NE, G * NB, G * MAXP, T, dev, ntc=_ntc, ninv=_ninv)
    my_hist = _scr["my_hist"]
    _hist_kernel[grid](tkf, my_hist, n_pairs, BLOCK=256)

    # 2. all_gather -> H[src, global_expert]  (1 collective, static size)
    # zeros, not empty, when the gather is skipped: build_offsets_fused still READS H for its
    # recv-side outputs (which are then overwritten from the transport), and uninitialised int32
    # feeding a cumsum makes those dead values nondeterministic run-to-run -- which would poison any
    # future buffer-level comparison exactly the way _scatter_kernel_hist's atomic_add already does.
    H = _h_buf(WORLD, NE, dev, zero_once=True)
    # else: H stays zeroed. build_offsets_fused's SEND-side outputs (send_counts, send_start, base)
    # read only my_hist and remain exact; its RECV-side outputs are overwritten in NcompassMoETransport.dispatch
    # from the signal + header rows. P3 removes the dead recv half of that kernel.


    # 3. per-group offsets. CAP mode: two fully-fused device kernels (NO D2H) replace the ~30 tiny
    # torch scan/searchsorted ops -> bit-identical (unit-verified). Legacy: torch scans + host lists.
    if r_cap is not None:
        # SLOTTED (L3): the recv buffer is WORLD*(P_CAP+1) rows (F2 -- the header row makes the slot
        # stride P_CAP+1), so recv ids / weights must cover the SAME row count as the buffer that
        # NcompassMoETransport._caps_for allocates. Env R_CAP (= WORLD*P_CAP) under-counts by exactly WORLD.
        if _SLOTTED and _cfg.p_cap():
            r_cap = WORLD * (_cfg.p_cap() + 1)
        (send_counts, recv_counts, send_start_gw, recv_base_gw, rev_recv_base_gw, base
         ) = build_offsets_fused(my_hist, H, ep_rank, WORLD, epg, E_LOCAL, G, dev,
                                 recv=not (_SLOTTED and _NOAG and _B2B))
        # L1: under the slotted layout the row->expert mapping must follow the SLOTS, not dense
        # source order, or every row is assigned the wrong expert.
        if _SLOTTED and _NOAG:
            # NcompassMoETransport.dispatch replaces m["recv_topk_ids"] wholesale with the header-derived ids
            # (built after the wait, when the headers have landed), so this is a placeholder that is
            # never read. Sentinel-filled rather than zeroed: 0 is a VALID expert id, so if the
            # replacement ever failed to happen we get masked-out rows, not silently wrong experts.
            recv_ids_all = _placeholder_ids(G, r_cap, epg, dev)
        elif _SLOTTED and _cfg.p_cap():
            recv_ids_all = build_recv_ids_slotted(H, ep_rank, r_cap, _cfg.p_cap(),
                                                  WORLD, epg, E_LOCAL, G, dev)   # [G,r_cap]
        else:
            recv_ids_all = build_recv_ids_fused(H, ep_rank, r_cap, WORLD, epg, E_LOCAL, G, dev)
        recv_w = _ones_col(r_cap, dev)
        counts_host = None
    else:
        mh = my_hist.view(WORLD, G, epg)                    # [dest, group, eig]
        Hv = H.view(WORLD, WORLD, G, epg)                   # [src, dest, group, eig]
        send_counts = mh.sum(2).t().contiguous()            # [G, WORLD]
        bucket_counts = mh.permute(1, 0, 2).reshape(G, NB)
        base = (bucket_counts.cumsum(1) - bucket_counts).to(torch.int32)
        recv_counts = Hv[:, ep_rank, :, :].sum(2).t().contiguous()   # [G, WORLD]
        std = Hv.sum(-1).to(torch.int64)                             # [src, dest, G]
        send_start_gw = (torch.cumsum(send_counts, 1) - send_counts).to(torch.int64)
        recv_base_dg = (torch.cumsum(std, 0) - std)[ep_rank]         # [dest, G]
        rev_recv_base_sg = (torch.cumsum(std, 1) - std)[:, ep_rank]  # [src, G]
        # The ONLY D2H -- kept ONLY for the NCCL a2a_v path (needs host split lists).
        counts_host = torch.stack([send_counts, recv_counts]).cpu()
        eig_ids = torch.arange(epg, device=dev, dtype=torch.int64).repeat(WORLD)

    # 4. scatter send rows into per-group buffers (both modes; needs `base`).
    # Dump-slot init: rows a group doesn't fill stay s_orig=T (extra x_ext/out_ext row), s_weight=0.
    if _scr is not None:
        counters, s_orig, s_weight = _scr["counters"], _scr["s_orig"], _scr["s_weight"]
    else:
        counters = torch.zeros(G * NB, dtype=torch.int32, device=dev)
        s_orig = torch.full((G * MAXP,), T, dtype=torch.int64, device=dev)
        s_weight = torch.zeros(G * MAXP, dtype=torch.float32, device=dev)
    _inv = _scr["inv_idx"] if (_FUSED_EPI and _scr is not None) else s_orig  # unused when INV=0
    _tctr = _scr["tok_ctr"] if (_FUSED_EPI and _scr is not None) else s_orig
    _scatter_kernel_hist[grid](tkf, twf, base.reshape(-1), counters, s_orig, s_weight,
                               _inv, _tctr, _TP1,
                               n_pairs, K, E_LOCAL, epg, WORLD, MAXP, NB, BLOCK=256,
                               INV=(1 if (_FUSED_EPI and _scr is not None) else 0))
    s_orig_v = s_orig.view(G, MAXP)
    s_weight_v = s_weight.view(G, MAXP)

    metas = []
    for g in range(G):
        if r_cap is not None:
            send_list = recv_list = None   # transport uses device counts, not host split lists
            P = R = 0                       # unused on the transport path (fully device-driven)
            recv_topk_ids = recv_ids_all[g].unsqueeze(1).contiguous()   # [r_cap,1]
            recv_topk_w = recv_w
            s_orig_m = s_orig_v[g].contiguous()
            s_weight_m = s_weight_v[g].unsqueeze(1).contiguous()
            recv_base_g = recv_base_gw[g].contiguous()
            rev_recv_base_g = rev_recv_base_gw[g].contiguous()
        else:
            send_list = counts_host[0, g].tolist()
            recv_list = counts_host[1, g].tolist()
            P = int(sum(send_list))
            R = int(sum(recv_list))
            # recv-expert-ids: repeat each (src,eig) bucket's group-relative id by its count.
            hsub = Hv[:, ep_rank, g, :].reshape(-1).to(torch.int64)      # [WORLD*epg] (src,eig)
            ids = torch.repeat_interleave(eig_ids, hsub, output_size=R)  # [R] in [0,epg)
            recv_topk_ids = ids.unsqueeze(1).contiguous()               # [R,1]
            recv_topk_w = torch.ones(R, 1, device=dev, dtype=torch.float32)
            s_orig_m = s_orig_v[g, :P].contiguous()
            s_weight_m = s_weight_v[g, :P].unsqueeze(1).contiguous()
            recv_base_g = recv_base_dg[:, g].contiguous()
            rev_recv_base_g = rev_recv_base_sg[:, g].contiguous()
        metas.append({
            "P": P, "R": R,
            "s_orig": s_orig_m,
            "s_weight": s_weight_m,
            "send_list": send_list, "recv_list": recv_list,
            "recv_topk_ids": recv_topk_ids,
            "recv_topk_w": recv_topk_w,
            # device offsets for the device-a2a transport (ncompass_transport); no host lists needed
            "send_counts_dev": send_counts[g].to(torch.int64).contiguous(),
            "recv_counts_dev": recv_counts[g].to(torch.int64).contiguous(),
            "send_start_dev": send_start_gw[g].contiguous(),
            "recv_base_dev": recv_base_g,
            "rev_recv_base_dev": rev_recv_base_g,
            # L3: the header row ships my per-expert counts to each dest so it can build recv ids
            # locally; expose the raw histogram + this meta's group index for that.
            "inv_idx": (_scr["inv_idx"].view(G, _TP1, K)[g]
                        if (_FUSED_EPI and _scr is not None) else None),
            "s_weight_flat": (s_weight.view(G, MAXP)[g] if _FUSED_EPI else None),
            "T_tokens": MAXP // K,
            "my_hist_dev": my_hist,
            "grp": g,
            "epg": epg,
            "e_local": E_LOCAL,
        })
    return metas


def build_group_meta(tk_ids, tk_w, E_LOCAL, epg, G, WORLD, comm, dev):
    """On-device counting-sort metadata for all G groups. Returns the same per-group `metas`
    list the pipeline consumes: {P, R, s_orig, s_weight, send_list, recv_list, recv_topk_ids,
    recv_topk_w}. `tk_ids` [T,K] global expert ids (long), `tk_w` [T,K] f32.
    """
    T, K = tk_ids.shape
    n_pairs = T * K
    MAXP = n_pairs                                   # a group holds at most all pairs
    tkf = tk_ids.reshape(-1).contiguous()
    twf = tk_w.reshape(-1).to(torch.float32).contiguous()

    send_counts = torch.zeros(G * WORLD, dtype=torch.int32, device=dev)
    grid = (triton.cdiv(n_pairs, 256),)
    _count_kernel[grid](tkf, send_counts, n_pairs, E_LOCAL, epg, WORLD, BLOCK=256)
    sc = send_counts.view(G, WORLD)
    base = (sc.cumsum(1) - sc).to(torch.int32)       # exclusive per-group dest offsets

    counters = torch.zeros(G * WORLD, dtype=torch.int32, device=dev)
    s_orig = torch.empty(G * MAXP, dtype=torch.int64, device=dev)
    s_weight = torch.empty(G * MAXP, dtype=torch.float32, device=dev)
    s_expert = torch.empty(G * MAXP, dtype=torch.int64, device=dev)
    _scatter_kernel[grid](tkf, twf, base.reshape(-1), counters,
                          s_orig, s_weight, s_expert,
                          n_pairs, K, E_LOCAL, epg, WORLD, MAXP, BLOCK=256)

    # count-exchange, ALL groups in one collective: send row i (G counts) to rank i; equal
    # (static) split -> no host lists. recv_wg[i, g] = pairs rank i sends me in group g.
    send_wg = sc.t().contiguous()                    # [WORLD, G]
    recv_wg = torch.empty_like(send_wg)
    dist.all_to_all_single(recv_wg, send_wg, group=comm)
    recv_counts = recv_wg.t().contiguous()           # [G, WORLD]

    # single irreducible D2H of the [2, G, WORLD] count block (for the a2a_v split lists)
    counts_host = torch.stack([sc, recv_counts]).cpu()
    s_orig_v = s_orig.view(G, MAXP)
    s_weight_v = s_weight.view(G, MAXP)
    s_expert_v = s_expert.view(G, MAXP)

    metas = []
    for g in range(G):
        send_list = counts_host[0, g].tolist()
        recv_list = counts_host[1, g].tolist()
        P = int(sum(send_list))
        R = int(sum(recv_list))
        s_expert_g = s_expert_v[g, :P].contiguous()
        recv_el = torch.empty(R, dtype=torch.int64, device=dev)
        dist.all_to_all_single(recv_el, s_expert_g, recv_list, send_list, group=comm)
        metas.append({
            "P": P, "R": R,
            "s_orig": s_orig_v[g, :P].contiguous(),
            "s_weight": s_weight_v[g, :P].unsqueeze(1).contiguous(),
            "send_list": send_list, "recv_list": recv_list,
            "recv_topk_ids": (recv_el - g * epg).unsqueeze(1).contiguous(),  # [R,1] in [0,epg)
            "recv_topk_w": torch.ones(R, 1, device=dev, dtype=torch.float32),
        })
    return metas


# =============================================================================================
# AGGRESSIVE FUSION (NCOMPASS_G2_HIST_FUSED=1). Collapse the ~20 tiny
# derivation ops + per-group Python loop into: _hist (reuse) -> all_gather -> _send_scan (K1) ->
# _scatter (reuse, K2) -> _recv_derive (K3) -> .cpu() -> _recv_ids (K4). Donors: vLLM
# moe_align_sum_kernels _moe_align_block_size + gpt-oss _routing_compute_expt_offs (K1 scan);
# megablocks MergeExpertPrefixSumKernel (K4 counts->ids). Global expert e = dest*E_LOCAL +
# group*epg + eig, E_LOCAL=G*epg, so my_hist/H.view(...,dest,group,eig) index directly.
# =============================================================================================


@triton.jit
def _send_scan_kernel(hist_ptr, send_counts_ptr, base_ptr,
                      E_LOCAL, epg, WORLD, NB,
                      BLOCK_NB: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_E: tl.constexpr):
    """K1: my_hist[NE] -> send_counts[G,WORLD] + base[G,NB]. One program per group (grid=(G,))."""
    g = tl.program_id(0)
    # base[g,b] = exclusive scan over (dest,eig) buckets b=dest*epg+eig of my_hist[dest,g,eig]
    b = tl.arange(0, BLOCK_NB)
    bmask = b < NB
    d = b // epg
    eig = b % epg
    bc = tl.load(hist_ptr + (d * E_LOCAL + g * epg + eig), mask=bmask, other=0)
    inc = tl.cumsum(bc, 0)
    tl.store(base_ptr + g * NB + b, inc - bc, mask=bmask)
    # send_counts[g,d] = sum over eig of my_hist[d,g,eig]
    wd = tl.arange(0, BLOCK_W)[:, None]
    we = tl.arange(0, BLOCK_E)[None, :]
    smask = (wd < WORLD) & (we < epg)
    bc2 = tl.load(hist_ptr + (wd * E_LOCAL + g * epg + we), mask=smask, other=0)
    sc = tl.sum(bc2, axis=1)
    wi = tl.arange(0, BLOCK_W)
    tl.store(send_counts_ptr + g * WORLD + wi, sc, mask=wi < WORLD)


@triton.jit
def _recv_derive_kernel(H_ptr, recv_counts_ptr, base_recv_ptr,
                        NE, E_LOCAL, epg, WORLD, ep_rank,
                        BLOCK_W: tl.constexpr, BLOCK_E: tl.constexpr):
    """K3: H[WORLD,NE] -> recv_counts[G,WORLD] + base_recv[G,WORLD] (exclusive scan over src).
    One program per group (grid=(G,))."""
    g = tl.program_id(0)
    s = tl.arange(0, BLOCK_W)[:, None]
    eig = tl.arange(0, BLOCK_E)[None, :]
    mask = (s < WORLD) & (eig < epg)
    Hsub = tl.load(H_ptr + (s * NE + ep_rank * E_LOCAL + g * epg + eig), mask=mask, other=0)
    rc = tl.sum(Hsub, axis=1)                       # [BLOCK_W]
    si = tl.arange(0, BLOCK_W)
    smask = si < WORLD
    tl.store(recv_counts_ptr + g * WORLD + si, rc, mask=smask)
    inc = tl.cumsum(rc, 0)
    tl.store(base_recv_ptr + g * WORLD + si, inc - rc, mask=smask)


@triton.jit
def _recv_ids_kernel(H_ptr, base_recv_ptr, ids_ptr, w_ptr,
                     NE, E_LOCAL, epg, WORLD, ep_rank, g,
                     MAXCNT: tl.constexpr, BLOCK: tl.constexpr, BLOCK_E: tl.constexpr):
    """K4: write group-relative recv-expert-ids (+ recv_topk_w=1) in source-major/eig-major order.
    One program per (src,eig) bucket (grid=(WORLD,epg)). Donor: megablocks MergeExpertPrefixSum."""
    s = tl.program_id(0)
    eig = tl.program_id(1)
    col0 = ep_rank * E_LOCAL + g * epg
    # eig_prefix = sum of H[s, col0 : col0+eig] (exclusive scan within source over eig)
    ee = tl.arange(0, BLOCK_E)
    prefix = tl.sum(tl.load(H_ptr + s * NE + col0 + ee, mask=ee < eig, other=0))
    count = tl.load(H_ptr + s * NE + col0 + eig)
    offset = tl.load(base_recv_ptr + g * WORLD + s) + prefix
    for i in range(0, MAXCNT, BLOCK):
        pos = i + tl.arange(0, BLOCK)
        m = pos < count
        tl.store(ids_ptr + offset + pos, eig, mask=m)
        tl.store(w_ptr + offset + pos, 1.0, mask=m)


def build_group_meta_hist_fused(tk_ids, tk_w, E_LOCAL, epg, G, WORLD, ep_rank, comm, dev):
    """Aggressively-fused build_group_meta_hist: same `metas` list, ~6 kernels + all_gather + 1 D2H.
    Correctness anchor = build_group_meta_hist / the AgRs oracle."""
    T, K = tk_ids.shape
    n_pairs = T * K
    MAXP = n_pairs
    NE = WORLD * E_LOCAL
    NB = WORLD * epg
    tkf = tk_ids.reshape(-1).contiguous()
    twf = tk_w.reshape(-1).to(torch.float32).contiguous()
    grid_p = (triton.cdiv(n_pairs, 256),)
    bw = triton.next_power_of_2(WORLD)
    be = triton.next_power_of_2(epg)

    my_hist = torch.zeros(NE, dtype=torch.int32, device=dev)
    _hist_kernel[grid_p](tkf, my_hist, n_pairs, BLOCK=256)
    if _SLOTTED:
        raise RuntimeError(
            "NCOMPASS_G2_SLOTTED=1 is incompatible with build_group_meta_hist_fused "
            "(NCOMPASS_G2_HIST_FUSED=1): it uses a dense exclusive scan and sizes recv ids to R "
            "rather than R_CAP, so it takes no r_cap at all (P0 audit B8). Use NCOMPASS_G2_HIST_META=1.")
    H = torch.empty(WORLD, NE, dtype=torch.int32, device=dev)
    dist.all_gather_into_tensor(H, my_hist.contiguous(), group=comm)

    send_counts = torch.empty(G, WORLD, dtype=torch.int32, device=dev)
    base = torch.empty(G, NB, dtype=torch.int32, device=dev)
    _send_scan_kernel[(G,)](my_hist, send_counts, base, E_LOCAL, epg, WORLD, NB,
                            BLOCK_NB=triton.next_power_of_2(NB), BLOCK_W=bw, BLOCK_E=be)

    counters = torch.zeros(G * NB, dtype=torch.int32, device=dev)
    s_orig = torch.empty(G * MAXP, dtype=torch.int64, device=dev)
    s_weight = torch.empty(G * MAXP, dtype=torch.float32, device=dev)
    _scatter_kernel_hist[grid_p](tkf, twf, base.reshape(-1), counters, s_orig, s_weight,
                                 n_pairs, K, E_LOCAL, epg, WORLD, MAXP, NB, BLOCK=256)
    s_orig_v = s_orig.view(G, MAXP)
    s_weight_v = s_weight.view(G, MAXP)

    recv_counts = torch.empty(G, WORLD, dtype=torch.int32, device=dev)
    base_recv = torch.empty(G, WORLD, dtype=torch.int32, device=dev)
    _recv_derive_kernel[(G,)](H, recv_counts, base_recv, NE, E_LOCAL, epg, WORLD, ep_rank,
                              BLOCK_W=bw, BLOCK_E=be)

    counts_host = torch.stack([send_counts, recv_counts]).cpu()   # the ONE D2H

    metas = []
    for g in range(G):
        send_list = counts_host[0, g].tolist()
        recv_list = counts_host[1, g].tolist()
        P = int(sum(send_list))
        R = int(sum(recv_list))
        recv_topk_ids = torch.empty(R, dtype=torch.int64, device=dev)
        recv_topk_w = torch.empty(R, dtype=torch.float32, device=dev)
        if R > 0:
            _recv_ids_kernel[(WORLD, epg)](H, base_recv, recv_topk_ids, recv_topk_w,
                                           NE, E_LOCAL, epg, WORLD, ep_rank, g,
                                           MAXCNT=1024, BLOCK=256, BLOCK_E=be)
        metas.append({
            "P": P, "R": R,
            "s_orig": s_orig_v[g, :P],              # already contiguous (own g*MAXP region)
            "s_weight": s_weight_v[g, :P].unsqueeze(1),
            "send_list": send_list, "recv_list": recv_list,
            "recv_topk_ids": recv_topk_ids.unsqueeze(1),
            "recv_topk_w": recv_topk_w.unsqueeze(1),
        })
    return metas
