# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ncompass: device-initiated all-to-all (dispatch/combine) for the G2 grouped-MoE pipeline.

Replaces `dist.all_to_all_single` (which needs HOST split-size lists from the metadata `.cpu()`)
with a device-count transport, removing the host split-list round-trip and making the dispatch
region CUDA-graph friendly. Validated correct intranode + cross-node world=16 and CUDA-graph
capturable with dynamic routing -- see ncompass/moe_pipeline/PLAN_cross_node_device_a2a_ibgda.md.

Mechanism (atomic-free, barrier-free, default IBRC transport): chunk-per-dest plain `nvshmem.put`
(data) -> `fence` -> 1-element plain put of `(2*ep+phase)*BIG + cnt+1` into the receiver's OWN
per-source counter slot (each source writes its own slot => no contention => no atomic) -> receiver
Triton spin-load poll until that value. `ep` = decode STEP index (host scalar, same on all ranks);
`phase` (0 dispatch / 1 combine) keeps a step's two exchanges distinct. Monotonic => no per-step
counter reset, no reset barrier, no NCCL interleaved with nvshmem in the hot path.

Offsets come from the metadata (build_group_meta_hist device tensors, derived from the already-
allgathered histogram) -> this module adds NO collectives. Tier 1: Marlin stays eager and consumes
the scalar `R = m["R"]` per group (from the metadata) as the recv row count.
"""

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from torch.distributed._symmetric_memory import _nvshmem_triton as nvshmem

from vllm.model_executor.layers.fused_moe import ncompass_moe_config as _cfg

_BIG = 1_000_000_000        # epoch stride in the flag value (> any per-step count)
# Verbose prints an "ok" line per call too, which is how a run proves the checker actually executed
# Q4 byte-volume measurement (eager only — uses .item()): mirror of all2all.py:_nc_log_moe_comm.
_LOGMOE_MAX = 3000
# Comm-kernel SM footprint: put/poll grid = NBLK blocks (default = WORLD, one per dest). Smaller NBLK
# frees SM slots so the transport co-resides with Marlin without evicting its CTAs (experiment: A/B).

# --- B5/B6 self-signalling slotted dispatch (optimization-spec 2026-08-06-transport-sentinel-slotted-
# dispatch rev 4, internal plan phases P1-P3). Both default OFF; with both off the legacy dense +
# fence + flag path below is bit-identical to before.
#   NCOMPASS_G2_SLOTTED  (L1): recv buffer = WORLD fixed slots of P_CAP rows, one contiguous
#                              right-aligned put per (source,dest) with a trailing header row.
#   NCOMPASS_G2_PUTSIG  (L2): fused put+signal -- ONE network op per peer carrying data AND
#                             signal (no fence, no separate 8B flag put), NIC-driven wait.
#
# HISTORY (do not resurrect): L2 was first built as a trailing SENTINEL whose epoch tag, at the
# highest address of the message, doubled as the arrival signal. That is UNDEFINED BEHAVIOUR under
# the NVSHMEM memory model -- a remote RMA payload write conflicting in parallel with a local
# spin-read; only AMOs and SIGNAL updates may be paired with wait/test. It passed every harness
# gate and measured a real -40.7 us, then deadlocked serve graph capture deterministically.
_SLOTTED = _cfg.SLOTTED
# S2 Phase-R oracle. Shadow buffers only -- never alters the live dispatch. Debug only.
# S2 rail-relay dispatch: cross each token the node boundary ONCE, fan out over NVLink.
_RELAY = _cfg.RELAY
# _relay_fanout copy shape. 8 warps x BLK_H=1024 is 4 bf16/thread = 8B stores (no 128-bit
# path) and 14 sequential passes per 7168-wide row. 32 warps x 8192 covers a row in ONE pass.
_RWARPS = _cfg.RELAY_WARPS
_RBLK = _cfg.RELAY_BLOCK
# _dispatch_put_sig block width. Its INTRA-node puts are a P2P copy performed by the calling
# block's own threads, so thread count sets how many NVLink requests are in flight; the rail/
# remote put only builds WQEs and is near-flat in it. Unlike _relay_fanout the vector width is
# NOT ours -- the copy lives inside NVSHMEM's putmem_signal_block -- so warps are the only knob.
#
# DEFAULT IS MODE-DEPENDENT, and the asymmetry is measured, not assumed:
#   MODE 0/1 (direct): 8 remote puts set the critical path and the 8 intra puts hide behind them,
#     so more threads speed up something off the critical path. Graphed sweep W=4/8/16/32 at sz=37
#     -> 0.802/0.804/0.802/0.804 ms, dispatch put 68.2/-/-/69.8 us. FLAT. Keep Triton's implicit
#     default of 4 so the shipped direct path stays byte-identical.
#   MODE 2 (relay): the rail collapses 8 remote puts into 1 (~40 us), which EXPOSES the intra puts.
#     At 4 warps they cost ~66 us and become the critical path, which is why the byte cut showed no
#     wall-time benefit. At 8 they drop below the rail term and Phase R hits the rail floor
#     (65.8 -> 38.8 us). Graphed, DIST=skew, relay-vs-direct: sz16 +2, sz37 +25, sz64 +71, sz90 +78,
#     sz128 +120 us -- the relay LOSES at every size at 4 warps and WINS at every size >=16 at 8.
#     Flat from 8 up (sz37: 0.775/0.786/0.778 at W=8/16/32), so 8 is the knee, not a peak.
_DWARPS = _cfg.DISPATCH_WARPS
# L3 P1b-2: consume the header-derived recv metadata instead of the AllGathered H.
_NOAG = _cfg.NOAG
_B2B = _cfg.B2B   # B2b metadata fusions
# SLOT GEOMETRY (L3, left-aligned) -- ONE definition; ncompass_routing_meta mirrors it via SLOT_STRIDE().
# Source s owns rows [s*STRIDE, s*STRIDE + P_CAP] in every destination's recv buffer:
#   data   = [s*STRIDE, s*STRIDE + cnt)      cnt data rows, left-aligned
#   header =  s*STRIDE + cnt                 ONE header row, immediately after the data
# so ONE contiguous put of (cnt+1) rows covers both -- required, because the NVSHMEM signal certifies
# delivery of ITS OWN dest data only (a plain put + a later put-with-signal implies nothing).
# STRIDE = P_CAP + 1, not P_CAP: the header consumes a row, so usable data capacity is P_CAP while a
# slot must physically hold cnt+1 <= P_CAP+1 rows. Sizing the recv buffer at WORLD*P_CAP (the old
# right-aligned assumption) silently corrupts slot s-1 when cnt == P_CAP, which DIST=hotspot reaches.
def SLOT_STRIDE(p_cap):
    return p_cap + 1     # C1/C7 slots -- DEFERRED (spec §5a)
# L2 (spec rev 6): fused put+signal transport. One network op per peer carrying data AND signal,
# no fence, no separate 8B flag put, NIC-driven wait. Dense legacy layout (no sentinel row).
# Donor: ncompass/moe_pipeline/dev_a2a_nvshmem.py:110-146.
_PUTSIG = _cfg.PUTSIG
# NVSHMEM enums, read from the SDK headers in the serve image -- do NOT guess these:
#   nvshmem_common_transport.h:30  typedef enum { NVSHMEM_SIGNAL_SET = 9, NVSHMEM_SIGNAL_ADD = 10 }
#   nvshmem_constants.h:47         NVSHMEM_CMP_EQ = 0
# Passing sig_op=0 (as the dev_a2a_nvshmem.py donor comment claims) is an INVALID op: the payload is
# still delivered but the signal is silently dropped -- observed as rel_l2 PASS with flags all zero,
# then an infinite wait. The donor gets away with it because it calls PyTorch's _nvshmem_triton
# binding, which translates the enum; ibgda_wrap is a raw passthrough.
_SIG_SET = tl.constexpr(9)
_CMP_EQ = tl.constexpr(0)
_CMP_GE = tl.constexpr(5)   # nvshmem_constants.h: EQ=0, NE, GT, LE, LT, GE -> 5


def _assert_slotted_preconditions():
    """The slotted put is built on the IBGDA wrapper (`_ib.putmem_block`), which only exists when
    NCOMPASS_G2_IBGDA=1. Fail with a clear message instead of a Triton trace-time NameError."""
    if not _IBGDA:
        raise RuntimeError(
            "NCOMPASS_G2_SLOTTED=1 requires NCOMPASS_G2_IBGDA=1: the slotted transport uses the "
            "IBGDA device wrapper. Enable IBGDA or disable NCOMPASS_G2_SLOTTED."
        )


# --- L1 IBGDA transport (NCOMPASS_G2_IBGDA=1): GPU-initiated puts via the wrapper+nvlink path
# (ibgda_shmem) instead of torch's IBGDA-free libnvshmem_device.bc. Eliminates the IBRC CPU proxy on
# the puts (~2.6x faster intra-node, ~1.7x aggregate dispatch at H=7168 8-NIC). Requires VMM-on + the
# IBGDA env in the serve config; the wrapper ships in-package (ncompass_ibgda/).
_IBGDA = _cfg.IBGDA
# L3 fused-overlap transport (Spec rev3): single _transport_ib kernel (nbi dispatch+wait overlapped) +
# deferred quiet. Requires IBGDA. ~2.25x transport vs sequential dispatch+wait in the standalone.
_ib = None
if _IBGDA:
    # In-tree since the single-flag cleanup; no sys.path insertion, no bind-mount. The
    # compiled wrapper object lives beside the module and is a build output (see
    # ncompass_ibgda/build_wrapper.sh); IBGDA_WRAPPER_O still overrides its location.
    from vllm.model_executor.layers.fused_moe.ncompass_ibgda import ibgda_shmem as _ib

    @triton.jit
    def _dispatch_put_ib(recv_ptr, send_ptr, cnt_ptr, sstart_ptr, rbase_ptr, ctr_base_ptr, myrank,
                         fsrc_ptr, ep_ptr, WORLD: tl.constexpr, H: tl.constexpr, ESZ: tl.constexpr,
                         PHASE: tl.constexpr, BIG: tl.constexpr, FUSED: tl.constexpr):
        # IBGDA twin of _dispatch_put: identical logic, transport via ib.* wrappers (byte sizes), plain
        # @triton.jit (NOT @requires_nvshmem -> avoids the IBGDA-free .bc; ibgda_shmem nvlinks the .a).
        pid = tl.program_id(0)
        nblk = tl.num_programs(0)
        ep2 = 2 * tl.load(ep_ptr) + PHASE
        for d in tl.static_range(WORLD):
            if (d % nblk) == pid:
                cnt = tl.load(cnt_ptr + d)
                if FUSED:
                    tl.store(fsrc_ptr + d, ep2 * BIG + cnt + 1)
                if cnt > 0:
                    so = tl.load(sstart_ptr + d)
                    rb = tl.load(rbase_ptr + d)
                    _ib.putmem_block(recv_ptr + rb * H, send_ptr + so * H, cnt * H * ESZ, d)
                _ib.fence()
                _ib.putmem_block(ctr_base_ptr + myrank, fsrc_ptr + d, cnt * 0 + 8, d)   # 8B int64 flag

    @triton.jit
    def _transport_ib(recv_ptr, send_ptr, cnt_ptr, sstart_ptr, rbase_ptr, ctr_ptr, myrank, fsrc_ptr,
                      expected_ptr, target_ptr, ep_ptr, WORLD: tl.constexpr, H: tl.constexpr,
                      ESZ: tl.constexpr, PHASE: tl.constexpr, BIG: tl.constexpr, FUSED: tl.constexpr):
        # L3 FUSED-OVERLAP (Spec rev3): one kernel = dispatch+wait. Post ALL my nbi sends (data then, after
        # one fence, flags) -- delivered async, NO per-call quiet -- then spin-wait ALL my incoming. My send
        # completion overlaps my recv-wait on the full-duplex NICs -> ~2.25x vs sequential dispatch+wait.
        # quiet is deferred to _quiet_ib before the next send-fill (send-buffer reuse).
        pid = tl.program_id(0)
        nblk = tl.num_programs(0)
        ep2 = 2 * tl.load(ep_ptr) + PHASE
        for d in tl.static_range(WORLD):                     # post all data (nbi)
            if (d % nblk) == pid:
                cnt = tl.load(cnt_ptr + d)
                if cnt > 0:
                    so = tl.load(sstart_ptr + d)
                    rb = tl.load(rbase_ptr + d)
                    _ib.putmem_nbi_block(recv_ptr + rb * H, send_ptr + so * H, cnt * H * ESZ, d)
        _ib.fence()                                          # order all data before all flags
        for d in tl.static_range(WORLD):                     # post all flags (nbi, plain)
            if (d % nblk) == pid:
                cnt = tl.load(cnt_ptr + d)
                if FUSED:
                    tl.store(fsrc_ptr + d, ep2 * BIG + cnt + 1)
                _ib.putmem_nbi_block(ctr_ptr + myrank, fsrc_ptr + d, ep2 * 0 + 8, d)
        for s in tl.static_range(WORLD):                     # spin-wait all incoming (overlaps my sends)
            if (s % nblk) == pid:
                exp = tl.load(expected_ptr + s)
                if exp > 0:
                    if FUSED:
                        tgt = ep2 * BIG + exp + 1
                    else:
                        tgt = tl.load(target_ptr + s)
                    val = tl.load(ctr_ptr + s, volatile=True)
                    while val != tgt:
                        val = tl.load(ctr_ptr + s, volatile=True)
        _ib.quiet()   # self-contained: drain MY nbi sends before return (~free -- done during the spin)



# ===================================================================================
# B5/B6 SLOTTED layout kernels (L1). Recv buffer = WORLD fixed slots of P_CAP rows; source s owns
# slot s in EVERY destination. Within its slot a source writes RIGHT-ALIGNED: cnt data rows
# immediately below a reserved header row that always sits at the slot's top row,
#   header_row(s) = (s+1)*P_CAP - 1,   data rows = [(s+1)*P_CAP - 1 - cnt, header_row)
# so ONE contiguous put of (cnt+1) rows always ends at the same address. That fixed end address is
# what L2 (P2) will poll; right-alignment is what makes the signal position independent of cnt.
#
# Send side must be contiguous [cnt data rows][header row] per dest, so the send buffer gains one
# header row per destination: dest d's region starts at send_start[d] + d, i.e. pair p of dest d
# gathers to send row (p + d). Send cap is therefore P_CAP + WORLD.
# tl.constexpr, not plain ints: @jit bodies may only close over constexpr globals (see _k_write_header).
_HDR_EPOCH_W = tl.constexpr(1)   # epoch tag: LAST int32 word of the header row (highest address)
_HDR_COUNT_W = tl.constexpr(0)   # total row count: first int32 word


@triton.jit
def _k_gather_slotted(x_ptr, send_ptr, s_orig_ptr, sstart_ptr, cnt_ptr,
                      send_i32_ptr, mh_ptr, ep_ptr, NPAIR,
                      WORLD: tl.constexpr, H: tl.constexpr, BLK_H: tl.constexpr,
                      SHIFT: tl.constexpr, IDX: tl.constexpr, HDR: tl.constexpr,
                      EPG_P: tl.constexpr = 1, epg: tl.constexpr = 0,
                      E_LOCAL: tl.constexpr = 0, GRP: tl.constexpr = 0,
                      PHASE: tl.constexpr = 0, BIG: tl.constexpr = 0,
                      RELAY: tl.constexpr = 0, GPN: tl.constexpr = 8,
                      MYNODE: tl.constexpr = 0, rail_ptr=None, rail_i32_ptr=None,
                      NTOK: tl.constexpr = 0):
    """Gather x[s_orig[p]] -> send[p + dest_of_pair(p)], i.e. the dense gather shifted by +d so each
    destination's group is followed by its own header row. dest is found by a 16-way scan over
    send_start/send_counts (block-uniform scalars, ~free) instead of a precomputed dest_of_pair."""
    p = tl.program_id(0)
    if HDR:
        # B2b-2: the header rows are INDEPENDENT of the gather -- they land at
        # sstart[d]+d+cnt[d], past each destination's data -- so they can ride this kernel's tail
        # instead of costing a second launch. Grid is max(NPAIR, WORLD): programs below WORLD write
        # a header, programs below NPAIR gather a row, and the two sets overlap harmlessly.
        if p < WORLD:
            d = p
            hcnt = tl.load(cnt_ptr + d)
            hss = tl.load(sstart_ptr + d)
            w32 = H // 2
            hbase = (hss + d + hcnt) * w32
            tl.store(send_i32_ptr + hbase + 0, hcnt.to(tl.int32))
            tl.store(send_i32_ptr + hbase + 1, hss.to(tl.int32))
            hep = tl.arange(0, EPG_P)
            hrealm = hep < epg
            hmh = tl.load(mh_ptr + d * E_LOCAL + GRP * epg + hep, mask=hrealm, other=0).to(tl.int32)
            tl.store(send_i32_ptr + hbase + 2 + hep, hmh, mask=hrealm)
            hep2 = 2 * tl.load(ep_ptr) + PHASE
            tl.store(send_i32_ptr + hbase + (w32 - 1), (hep2 * BIG + hcnt + 1).to(tl.int32))
            if RELAY:
                # Rail payload tail. The relay cannot reconstruct these: the receiver's
                # _k_recv_ids_from_hdr reads a HEADER ROW per source, and the relay has no access to
                # the partner's my_hist. So the header rows ride along, one per remote dest, plus a
                # meta row carrying cnt/sstart so the relay can slice the descriptor list.
                # Written from the SAME values as the send-side header above -- if these two ever
                # diverge the receiver decodes a different cnt than the signal says.
                rw = H // 2
                tl.store(rail_i32_ptr + (NTOK + 1) * rw + d, hcnt.to(tl.int32))
                tl.store(rail_i32_ptr + (NTOK + 1) * rw + WORLD + d, hss.to(tl.int32))
                if (d // GPN) != MYNODE:
                    rb2 = (NTOK + 2 + (d - (1 - MYNODE) * GPN)) * rw
                    tl.store(rail_i32_ptr + rb2 + 0, hcnt.to(tl.int32))
                    tl.store(rail_i32_ptr + rb2 + 1, hss.to(tl.int32))
                    tl.store(rail_i32_ptr + rb2 + 2 + hep, hmh, mask=hrealm)
                    tl.store(rail_i32_ptr + rb2 + (rw - 1),
                             (hep2 * BIG + hcnt + 1).to(tl.int32))
    if HDR:
        if p >= NPAIR:
            return
    # IDX=1: src is the FULL x and row p is x[s_orig[p]] (the fused L4 gather). IDX=0: src is ALREADY
    # gathered dense ([MAXP,H]) and row p is simply src[p] -- the non-fused dispatch path, which must
    # still receive the +d shift or the put (which always reads sstart[d]+d) reads the wrong rows.
    if RELAY:
        # Phase R's payload rows. NTOK <= NPAIR always (NPAIR = T*top_k), so these programs exist.
        # Row index is the TOKEN index, not the pair index, so the relay can address it straight
        # from the descriptor (which carries s_orig[p]) with no extra indirection table.
        # This is a copy into the symmetric heap, not a gather: nvshmem needs the put source there,
        # and x/x_ext is an ordinary tensor. One store per program for the first NTOK programs --
        # folded here rather than paid as a launch (spec rev 10 §7 kernel-count constraint).
        if p < NTOK:
            for roff in tl.static_range(0, H, BLK_H):
                ridx = roff + tl.arange(0, BLK_H)
                rmm = ridx < H
                tl.store(rail_ptr + p * H + ridx,
                         tl.load(x_ptr + p * H + ridx, mask=rmm), mask=rmm)
    t = tl.load(s_orig_ptr + p) if IDX else p
    # dest index of pair p, BRANCH-FREE. The previous form reassigned a scalar under a runtime
    # predicate inside a static_range; that lowers to a loop-carried select and was miscomputing `d`,
    # which made two pairs collide on one row -- losing and duplicating rows while leaving the row
    # COUNT and the contiguous prefix intact, so the shape checks could not see it.
    # Destination ranges are contiguous and ordered, so d = #{j : p >= send_start[j] + cnt[j]}.
    # Empty destinations are counted correctly (they still consume a sentinel row).
    sv = tl.arange(0, WORLD)
    ssv = tl.load(sstart_ptr + sv)
    ccv = tl.load(cnt_ptr + sv)
    d = tl.sum((p >= ssv + ccv).to(tl.int32), axis=0)
    if RELAY:
        # S2 rail-relay descriptor. The relay re-derives `d` and the slot position from the SAME
        # branch-free scan above (it receives cnt/sstart in the Phase-R header), so the only per-pair
        # payload it cannot reconstruct is which token row to read -- that is `t`.
        #
        # NO COMPACTION. The retired hierarchical build stream-compacted unique remote tokens
        # with atomics. S0 measured unique_remote_tokens == T EXACTLY at every sweep point
        # -- 1-(1/2)^top_k = 99.6% of tokens have at least one remote expert -- so the whole token
        # block IS the deduplicated set to within 0.4%, and Phase R can send x[0:T] directly. That
        # deletes the atomics, the counter, and a kernel.
        if p < NPAIR:
            is_rem = (d // GPN) != MYNODE
            # descriptors ride IN the rail payload (row NTOK) so Phase R ships them in its one put
            tl.store(rail_i32_ptr + NTOK * (H // 2) + p, tl.where(is_rem, t, -1).to(tl.int32))
    if t >= 0:
        out = p + d if SHIFT else p
        for off in tl.static_range(0, H, BLK_H):
            idx = off + tl.arange(0, BLK_H)
            mm = idx < H
            tl.store(send_ptr + out * H + idx, tl.load(x_ptr + t * H + idx, mask=mm), mask=mm)


@triton.jit
def _k_write_header(send_i32_ptr, cnt_ptr, sstart_ptr, mh_ptr, ep_ptr,
                    WORLD: tl.constexpr, H: tl.constexpr, EPG_P: tl.constexpr, epg: tl.constexpr,
                    E_LOCAL: tl.constexpr, GRP: tl.constexpr, PHASE: tl.constexpr, BIG: tl.constexpr):
    """Header row for dest d, at send row (send_start[d] + d + cnt[d]) -- immediately after d's data,
    so ONE put of cnt+1 rows carries both.

    Payload (int32 words in an H-bf16 = H/2-int32 row):
        [0]                  cnt                    -- rows in this message
        [1]                  send_start[d]          -- MY send_start for d == d's `rev_recv_base[me]`,
                                                      i.e. where d must return my rows. d cannot derive
                                                      this: it is MY local view of MY sends to ranks
                                                      before d. Shipping it here is what lets P1b drop
                                                      the AllGather entirely rather than only its
                                                      recv_base role.
        [2 .. 2+epg)         my_hist[d, g*epg+eig]  -- per-expert counts, so d builds recv ids locally
        [w-1]                epoch tag              -- retained for debug/versioning only; the arrival
                                                      signal is the flags pad (put+signal), NOT this.
    Written in P1a but NOT yet consumed (ids still come from H); P1b consumes it.
    """
    d = tl.program_id(0)
    if d < WORLD:
        cnt = tl.load(cnt_ptr + d)
        ss = tl.load(sstart_ptr + d)
        w = H // 2
        base = (ss + d + cnt) * w
        tl.store(send_i32_ptr + base + 0, cnt.to(tl.int32))
        tl.store(send_i32_ptr + base + 1, ss.to(tl.int32))
        ep = tl.arange(0, EPG_P)
        realm = ep < epg
        mh = tl.load(mh_ptr + d * E_LOCAL + GRP * epg + ep, mask=realm, other=0).to(tl.int32)
        tl.store(send_i32_ptr + base + 2 + ep, mh, mask=realm)
        ep2 = 2 * tl.load(ep_ptr) + PHASE
        tl.store(send_i32_ptr + base + (w - 1), (ep2 * BIG + cnt + 1).to(tl.int32))



# --- C2 (spec rev 5): self-signalling dispatch with the DENSE layout retained. -------------------
# Each source's block gains ONE trailing sentinel row, so the single contiguous put ends on it and its
# epoch tag is the last word to become visible. The receiver knows every source's base (exclusive
# cumsum of recv_counts, computed LOCALLY) and count, so the sentinel address is computable -- no slots
# and no AllGather removal. Valid rows stay a contiguous PREFIX, which is what keeps the downstream
# consumer chain intact (the invariant lever L1/slots destroyed; see spec §5a).
#
# The tag must fit int32 (it lives in a bf16 row reinterpreted as int32), so the epoch stride is 2^16
# rather than _BIG: tag = (ep2 % 32768)*65536 + cnt + 1, whose max is exactly INT32_MAX.
# Triton @jit bodies may only close over globals declared as tl.constexpr (a plain int raises
# NameError at compile time). These are referenced from _k_write_sentinel / _wait_sentinel only.


@triton.jit
def _dispatch_put_slotted(recv_ptr, send_ptr, cnt_ptr, sstart_ptr, rbase_ptr, ctr_base_ptr, myrank,
                          fsrc_ptr, ep_ptr, WORLD: tl.constexpr, H: tl.constexpr, ESZ: tl.constexpr,
                          PHASE: tl.constexpr, BIG: tl.constexpr, P_CAP: tl.constexpr,
                          MODE: tl.constexpr):
    """Slot-addressed twin of `_dispatch_put_ib`. Signalling is UNCHANGED (fence + 8B flag) so P1
    validates the LAYOUT in isolation; P2 replaces the signalling only.

    RETIRED: superseded by `_dispatch_put_sig` with SLOT=1, which carries the same LEFT-aligned
    slot addressing over put+signal. Unreachable -- `_SLOTTED` now requires `_PUTSIG` (see __init__).
    Kept only until P1b removes the fence+flag pathway. Its addressing below is the OLD
    RIGHT-aligned convention and does NOT match the shipped layout."""
    pid = tl.program_id(0)
    nblk = tl.num_programs(0)
    ep2 = 2 * tl.load(ep_ptr) + PHASE
    for d in tl.static_range(WORLD):
        if (d % nblk) == pid:
            cnt = tl.load(cnt_ptr + d)
            tl.store(fsrc_ptr + d, ep2 * BIG + cnt + 1)
            if MODE == 0:
                dst_row = (myrank + 1) * P_CAP - 1 - cnt
                src_row = tl.load(sstart_ptr + d) + d
                nrow = cnt + 1                        # data + header, one contiguous message
            else:
                dst_row = tl.load(rbase_ptr + d)
                src_row = (d + 1) * P_CAP - 1 - cnt
                nrow = cnt
            if nrow > 0:
                _ib.putmem_block(recv_ptr + dst_row * H, send_ptr + src_row * H,
                                 nrow * H * ESZ, d)
            _ib.fence()
            _ib.putmem_block(ctr_base_ptr + myrank, fsrc_ptr + d, cnt * 0 + 8, d)


@nvshmem.requires_nvshmem
@triton.jit
def _dispatch_put(recv_ptr, send_ptr, cnt_ptr, sstart_ptr, rbase_ptr, ctr_base_ptr, myrank,
                  fsrc_ptr, ep_ptr, WORLD: tl.constexpr, H: tl.constexpr,
                  PHASE: tl.constexpr, BIG: tl.constexpr, FUSED: tl.constexpr):
    # grid=(NBLK,), NBLK<=WORLD: each block grid-strides over the dests it owns, issuing each dest's
    # (async RDMA data put -> fence -> 1-elem flag put). NBLK<WORLD frees SM slots for Marlin overlap;
    # the puts are async (SM only issues the descriptor) so fewer blocks ~= same latency.
    # FUSED (L2): compute the flag value (2*ep+phase)*BIG+cnt+1 IN-KERNEL from the device epoch and the
    # device count, storing it into fsrc[d] before the data put -> removes the eager fsrc.copy_ chain
    # (~3 tiny long-arith kernels per _run). The store precedes the data-put+fence, so fsrc[d] is
    # committed well before the flag put reads it.
    pid = tl.program_id(0)
    nblk = tl.num_programs(0)
    ep2 = 2 * tl.load(ep_ptr) + PHASE
    for d in tl.static_range(WORLD):
        if (d % nblk) == pid:
            cnt = tl.load(cnt_ptr + d)
            if FUSED:
                tl.store(fsrc_ptr + d, ep2 * BIG + cnt + 1)
            if cnt > 0:
                so = tl.load(sstart_ptr + d)
                rb = tl.load(rbase_ptr + d)
                nvshmem.put(recv_ptr + rb * H, send_ptr + so * H, cnt * H, d)
            nvshmem.fence()
            nvshmem.put(ctr_base_ptr + myrank, fsrc_ptr + d, 1, d)   # flag -> dest d's counter[myrank]


@triton.jit
def _dispatch_put_sig(recv_ptr, send_ptr, cnt_ptr, sstart_ptr, rbase_ptr, ctr_base_ptr, myrank,
                      fsrc_ptr, ep_ptr, WORLD: tl.constexpr, H: tl.constexpr, ESZ: tl.constexpr,
                      PHASE: tl.constexpr, BIG: tl.constexpr, FUSED: tl.constexpr,
                      SLOT: tl.constexpr, STRIDE: tl.constexpr, MODE: tl.constexpr,
                      land_ptr=None, rail_ptr=None, arr_ptr=None, scr_ptr=None,
                      GPN: tl.constexpr = 8, NTOK: tl.constexpr = 0,
                      RGUARD: tl.constexpr = 0, RDIAG: tl.constexpr = 0):
    """Fused data+signal put: ONE network op per peer (no fence, no separate 8B flag put).

    GRID IS (WORLD,) AND `d = program_id` DELIBERATELY -- do NOT reshape this into the
    `for d in static_range(WORLD): if (d % nblk) == pid` form used by `_dispatch_put_ib`.
    `nvshmemx_putmem_signal_block` is a BLOCK-SCOPED collective that must __syncthreads() internally to
    order the payload ahead of the signal; replicating it as 16 predicated unrolled copies hangs
    (observed: harness wedged for the full 420 s timeout). Plain `putmem_block` tolerates that shape,
    a put-with-signal does not. Donor: ncompass/moe_pipeline/dev_a2a_nvshmem.py:113-131.

    SPEC CONSTRAINT (NVSHMEM signal.md): the signal certifies delivery of ITS OWN `dest` data only, so
    a peer's payload must ride in THIS single op -- never a plain put followed by a put-with-signal.
    `sig_addr` (flags pad) and `dest` (recv) are separate allocations, as the spec requires.

    Addressing is the LEGACY DENSE one; the signal lives in the flags pad, so no layout change.
    cnt==0 dests are not signalled and `_wait_sig` skips exp==0 sources (symmetric: my cnt[d] == d's
    exp[me]).
    """
    d = tl.program_id(0)
    cnt = tl.load(cnt_ptr + d)
    ep2 = 2 * tl.load(ep_ptr) + PHASE
    if FUSED:
        tl.store(fsrc_ptr + d, ep2 * BIG + cnt + 1)
    # SLOT and MODE are tl.constexpr, so every branch below is resolved at COMPILE time and each
    # specialisation contains exactly ONE unconditional `putmem_signal_block`. Do not collapse these
    # into a single `if SLOT and MODE == 0:` / `elif cnt > 0:` chain: if any part of the predicate
    # were ever lowered to a RUNTIME branch it would wrap a block-scoped collective, which is the
    # documented hang (see this function's grid note).
    if SLOT:
        if MODE == 0:
            # DISPATCH, slotted: my rows always land in slot `myrank` of d's recv buffer, so the
            # destination address needs NO knowledge of other sources' counts -- this is what deletes
            # the metadata AllGather. Send side is [data(cnt)][header] contiguous at sstart[d]+d (the
            # +d shift the gather already applies), so nrow = cnt+1 puts both in ONE op, which the
            # signal's delivery guarantee requires.
            #
            # UNCONDITIONAL -- deliberately NO `if cnt > 0`. Once the AllGather is gone the receiver
            # cannot know which sources are empty, so an unsignalled empty source would either
            # deadlock the wait or be skipped while a non-empty one is read stale. Always sending the
            # (cnt+1)-row block makes every source signal every epoch, so `expected` collapses to the
            # constant WORLD and the receiver recovers cnt from the signal VALUE (_wait_sig_slot).
            # Cost when cnt == 0: one 1-row put (H*ESZ bytes), rare in decode.
            sv = tl.load(fsrc_ptr + d)                 # epoch-tagged: ep2*BIG + cnt + 1
            so = tl.load(sstart_ptr + d) + d
            _ib.putmem_signal_block(recv_ptr + (myrank * STRIDE) * H, send_ptr + so * H,
                                    (cnt + 1) * H * ESZ,
                                    ctr_base_ptr + myrank, sv, _SIG_SET, d)   # sig_op MUST be 9
        elif MODE == 2:
            # S2 PHASE R -- rail-relay dispatch. Three roles by `d`, ONE call site:
            #   intra-node d   -> the normal MODE 0 slotted put, unchanged
            #   d == myrank^GPN -> the rail peer: deduplicated token block + descriptors + header,
            #                      into its LANDING buffer, signalling its ARRIVAL counter
            #   other remote d -> INACTIVE: its rows are delivered by the relay, not by us
            #
            # The three cases SELECT arguments instead of branching. Wrapping a block-scoped
            # collective in a runtime predicate is the documented hang (grid note above), and the
            # 16-predicated-copies form is exactly what wedged the harness for 420 s. One
            # unconditional putmem_signal_block, every operand a tl.where.
            #
            # Inactive dests are made harmless by nbytes=0, pe=myrank (self, so nothing leaves the
            # GPU) and a signal redirected to local scratch -- NOT by skipping the call. Signalling
            # ctr_base[myrank] on a remote dest here would be a lie: the receiver would see source
            # `myrank` complete before the relay had written its rows.
            paired = myrank ^ GPN
            is_intra = (d // GPN) == (myrank // GPN)
            is_pair = d == paired
            active = is_intra | is_pair
            i_dst = recv_ptr.to(tl.int64) + (myrank * STRIDE) * H * ESZ
            i_src = send_ptr.to(tl.int64) + (tl.load(sstart_ptr + d) + d) * H * ESZ
            i_n = (cnt + 1) * H * ESZ
            # rail payload is contiguous by construction:
            #   [NTOK tokens][desc row][meta row][GPN forwarded header rows]
            # NTOK here is the SENDER's own count, which is correct -- it sizes its own payload.
            # It used to read "NTOK is a constexpr (the captured graph size)", which was the
            # load-bearing (and unenforced) assumption behind this transport's worst bug: the RELAY
            # located fields in this payload using ITS OWN NTOK. True under capture, where every rank
            # replays the same shape; false on the eager path, where rail partners are `rank ^ GPN`
            # -- DIFFERENT DP groups whose counts match only by padding. The count now travels in the
            # arrival signal below and the receiver derives every offset from it.
            r_n = (NTOK + 2 + GPN) * H * ESZ
            dst = tl.where(is_pair, land_ptr.to(tl.int64), i_dst)
            src = tl.where(is_pair, rail_ptr.to(tl.int64), i_src)
            nb = tl.where(active, tl.where(is_pair, r_n, i_n), 0)
            sg = tl.where(is_pair, arr_ptr.to(tl.int64) + myrank * 8,
                          ctr_base_ptr.to(tl.int64) + myrank * 8)
            sg = tl.where(active, sg, scr_ptr.to(tl.int64))
            pe = tl.where(active, d, myrank)
            # RDIAG (timing-only, breaks data; signals still fire so the pipeline completes):
            #   1 = rail bytes only (intra zeroed)   2 = intra bytes only (rail zeroed)
            if RDIAG == 1:
                nb = tl.where(is_pair, nb, 0)
            if RDIAG == 2:
                nb = tl.where(is_pair, 0, nb)
            # L1: the rail arrival signal CARRIES THIS SENDER'S TOKEN COUNT, `ep2*BIG + NTOK + 1`.
            # The relay parses a payload written HERE but was previously locating its fields with the
            # RELAY's own NTOK -- correct only while both ranks carry the same count. Rail partners
            # are `rank ^ GPN`, i.e. DIFFERENT DP groups (EP4=DP1 <-> EP12=DP3), whose counts coincide
            # only by DP padding, so the assumption is silently load-bearing. When it broke, the
            # relay read token rows as a descriptor and dereferenced the resulting garbage row index.
            # Encoding the count here is the same idiom `_wait_sig_slot` already uses for the
            # dispatch wait; the receiver decodes it and derives every offset from it.
            sv = tl.where(is_pair, ep2 * BIG + NTOK + 1, tl.load(fsrc_ptr + d))
            # DIAG (RGUARD): skip the inactive blocks' put entirely instead of neutralising it with
            # nbytes=0. `active` depends only on `d`, so it is BLOCK-UNIFORM -- the same shape as the
            # `if cnt > 0:` guard MODE 1 already ships, not the 16-predicated-copies form that hangs.
            if RGUARD:
                if active:
                    _ib.putmem_signal_block(dst.to(tl.pointer_type(tl.bfloat16)),
                                            src.to(tl.pointer_type(tl.bfloat16)), nb,
                                            sg.to(tl.pointer_type(tl.int64)), sv, _SIG_SET, pe)
            else:
                _ib.putmem_signal_block(dst.to(tl.pointer_type(tl.bfloat16)),
                                        src.to(tl.pointer_type(tl.bfloat16)), nb,
                                        sg.to(tl.pointer_type(tl.int64)), sv, _SIG_SET, pe)
        else:
            # COMBINE: read back source d's block from MY slot d, return `cnt` data rows (no header).
            # Still guarded on cnt: the combine's `expected` is the dispatch's own send_counts, which
            # this rank computed locally and therefore always has -- no AllGather dependence to break.
            if cnt > 0:
                sv = tl.load(fsrc_ptr + d)
                _ib.putmem_signal_block(recv_ptr + tl.load(rbase_ptr + d) * H,
                                        send_ptr + (d * STRIDE) * H, cnt * H * ESZ,
                                        ctr_base_ptr + myrank, sv, _SIG_SET, d)
    else:
        if cnt > 0:
            sv = tl.load(fsrc_ptr + d)
            _ib.putmem_signal_block(recv_ptr + tl.load(rbase_ptr + d) * H,
                                    send_ptr + tl.load(sstart_ptr + d) * H, cnt * H * ESZ,
                                    ctr_base_ptr + myrank, sv, _SIG_SET, d)


@triton.jit
def _relay_fanout(recv_ptr, land_ptr, land_i32_ptr, stage_ptr, ctr_base_ptr, arr_ptr, ep_ptr,
                  scr_ptr, myrank, WORLD: tl.constexpr, GPN: tl.constexpr, H: tl.constexpr,
                  ESZ: tl.constexpr, STRIDE: tl.constexpr, NTOK: tl.constexpr,
                  PHASE: tl.constexpr, BIG: tl.constexpr, SCAP: tl.constexpr,
                  BLK_H: tl.constexpr = 1024, NOWAIT: tl.constexpr = 0):
    """S2 PHASE N -- the relay. THE ONE NEW KERNEL in the rail-relay design.

    I relay for my rail partner `myrank ^ GPN`: its Phase R put landed its whole token block in MY
    landing buffer, and I fan those rows out to my node-local dests.

    grid=(GPN,), ONE BLOCK PER LOCAL DEST (SPLIT=1). Deliberate for S2: with several blocks per dest
    the destination's flag must not be set until ALL of them have finished, which needs a cross-block
    barrier -- a second kernel, or plain peer stores plus a __threadfence_system visibility protocol.
    S2 is the correctness anchor and is expected to lose on wall time; SPLIT and the visibility
    protocol are S3's problem.

    Transport is `putmem_signal_block`, NOT plain peer stores. A raw st.global to a peer followed by
    a flag store needs system-scope fencing that the wrapper does not expose (`ib.fence()` orders
    NVSHMEM RMA, not plain stores). Reusing the direct path's put-with-signal gets data->signal
    ordering for free; the price is staging each dest's rows contiguously first, which is why
    `stage` exists and why there is a debug_barrier before the put.

    THE ADDRESSING INVARIANT, and the whole reason recv stays byte-identical to direct: I write into
    dest d's recv at slot **`partner`** -- the ORIGINAL source -- not at my own slot, and I set
    `flags[partner]` there, not `flags[myrank]`. The receiver cannot tell a relayed block from a
    direct one, so `_wait_sig_slot`, `_k_recv_ids_from_hdr` and the whole combine leg are untouched.
    """
    j = tl.program_id(0)                       # local dest index within my node
    mynode = myrank // GPN
    dst = mynode * GPN + j
    partner = myrank ^ GPN
    ep2 = 2 * tl.load(ep_ptr) + PHASE
    # Phase R's arrival. Epoch-tagged, so the flags pad needs no per-call zeroing.
    # NOWAIT (timing-only, RDIAG=3): skip the arrival wait to separate "waiting for Phase R to
    # land" from the relay's own copy+put work. Reads possibly-unlanded data -> rel_l2 will fail;
    # that is expected and must not be read as a correctness result.
    # L1: wait, then DECODE THE SENDER'S TOKEN COUNT out of the arrival signal and use it -- not the
    # local NTOK -- for every offset into the partner's payload.
    #
    # CMP_GE rather than CMP_EQ, matching `_wait_sig_slot`: the value now carries a count the receiver
    # cannot predict, so no equality target exists. The staleness half of that kernel's argument
    # transfers unchanged -- a value from epoch ep2-1 is at most (ep2-1)*BIG + SCAP + 1, below
    # ep2*BIG + 1 because NTOK+1 (<= 721) << BIG (1e9).
    #
    # The future-epoch half does NOT transfer for free, and that is why the range check below exists.
    # `_wait_sig_slot` can argue a future epoch cannot have landed because that wait gates the
    # sender's own next round; this arrival signal is written by a peer on the OTHER node whose
    # progress this wait does not directly gate. Under CMP_EQ a future-epoch value failed LOUDLY (the
    # wait simply never matched). Under CMP_GE it would be accepted silently. The design already
    # requires that case to be impossible -- `land` is single-buffered, so a partner a full epoch
    # ahead would corrupt the payload whatever the comparison -- but "already assumed impossible" is
    # exactly the class of unenforced assumption that caused this bug, so it is now CHECKED: a
    # future-epoch value decodes to ntok_src >= BIG and is caught here rather than followed.
    # int64 throughout: the decode below produces int64, and Triton requires every assignment to a
    # name to unify. A bare `ntok_src = NTOK` here (or in the guard) is int32 and fails to compile
    # with "initial value ... is of type int64[], but the then block redefines it as int32[]".
    ntok_src = tl.full((), NTOK, tl.int64)
    if NOWAIT == 0:
        _ib.signal_wait_until(arr_ptr + partner, _CMP_GE, ep2 * BIG + 1)
        av = tl.load(arr_ptr + partner)
        ntok_src = av - ep2 * BIG - 1
        if (ntok_src < 0) | (ntok_src > SCAP):
            # Contained + surfaced, never silently clamped: record for the host and fall back to the
            # local count so the kernel completes instead of dereferencing a wild offset.
            tl.atomic_max(scr_ptr + 8, ntok_src)
            tl.atomic_add(scr_ptr + 9, 1)
            ntok_src = tl.full((), NTOK, tl.int64)

    rw = H // 2
    # CLAMP: cnt_d comes off the wire. A corrupt/unlanded meta row would otherwise drive the staging
    # loop past the stage buffer and spin the GPU until the harness timeout -- which is exactly how
    # the NOWAIT and intra-only diagnostics wedged the cluster and left orphaned containers that
    # silently poisoned every subsequent measurement. Bound it to what the slot can physically hold.
    cnt_d = tl.minimum(tl.load(land_i32_ptr + (ntok_src + 1) * rw + dst).to(tl.int64), STRIDE - 1)
    sst_d = tl.load(land_i32_ptr + (ntok_src + 1) * rw + WORLD + dst).to(tl.int64)

    # L2 (defence in depth, NOT the fix). `cnt_d` above has always been clamped -- with a comment
    # recording that an unclamped count once wedged the cluster -- while `sst_d` and the per-row
    # index `t` were read off the wire and used raw, and `t` is dereferenced as `land_ptr + t*H`.
    # That asymmetry is what turned a layout disagreement into an illegal access instead of a
    # detectable error. With L1 the layout contract is explicit, so a violation here means a NEW
    # defect: it is recorded for the host (scr[10] worst value, scr[11] count) and the offending
    # work is skipped. It is never silently clamped into plausible-looking output -- this cluster
    # cannot detect wrong output, so silent corruption is strictly worse than a loud stop.
    # These branches contain no block-scoped collective, so they are not the predicated-put shape
    # the MODE-2 comments warn about.
    if (sst_d < 0) | (sst_d + cnt_d > ntok_src * rw):
        tl.atomic_max(scr_ptr + 10, sst_d)
        tl.atomic_add(scr_ptr + 11, 1)
        # int64, like every other assignment in this kernel -- a bare `0` is int32 and fails
        # unification exactly as the ntok_src fallback did.
        cnt_d = tl.full((), 0, tl.int64)   # stage nothing; the header row below still ships

    # stage [data(cnt_d)][header] contiguously, mirroring the send-side layout MODE 0 puts
    i = 0
    while i < cnt_d:
        t = tl.load(land_i32_ptr + ntok_src * rw + sst_d + i).to(tl.int64)
        if (t < 0) | (t >= ntok_src):
            tl.atomic_max(scr_ptr + 10, t)
            tl.atomic_add(scr_ptr + 11, 1)
            t = tl.full((), 0, tl.int64)   # contained; the host surfaces scr[11] loudly
        for off in tl.static_range(0, H, BLK_H):
            idx = off + tl.arange(0, BLK_H)
            mm = idx < H
            tl.store(stage_ptr + (j * STRIDE + i) * H + idx,
                     tl.load(land_ptr + t * H + idx, mask=mm), mask=mm)
        i += 1
    # the partner's own header row for this dest, forwarded verbatim by Phase R
    for hoff in tl.static_range(0, H, BLK_H):
        hidx = hoff + tl.arange(0, BLK_H)
        hmm = hidx < H
        tl.store(stage_ptr + (j * STRIDE + cnt_d) * H + hidx,
                 tl.load(land_ptr + (ntok_src + 2 + j) * H + hidx, mask=hmm), mask=hmm)
    tl.debug_barrier()                          # stage writes -> block-scoped put read

    # UNCONDITIONAL, exactly as MODE 0: every source signals every epoch even at cnt==0 (one header
    # row), so the receiver's `expected` stays the constant WORLD and it recovers cnt from the value.
    _ib.putmem_signal_block(recv_ptr + (partner * STRIDE) * H,
                            stage_ptr + (j * STRIDE) * H, (cnt_d + 1) * H * ESZ,
                            ctr_base_ptr + partner, ep2 * BIG + cnt_d + 1, _SIG_SET, dst)


@triton.jit
def _wait_sig(ctr_base_ptr, target_ptr, expected_ptr, ep_ptr, WORLD: tl.constexpr,
              PHASE: tl.constexpr, BIG: tl.constexpr, FUSED: tl.constexpr):
    """NIC-driven wait on the signal object -- no SM spin. grid=(WORLD,), one source per block
    (donor form). Legal per the NVSHMEM memory model: signal update + wait/test is an explicit
    exception to the conflicting-access rule that made the payload-embedded sentinel UB.
    Target matches `_wait_poll` exactly: ep2*BIG + expected + 1 (epoch-tagged, so the flags pad needs
    no per-call zeroing -- unlike the donor, which zeroes + barriers every iteration)."""
    s = tl.program_id(0)
    exp = tl.load(expected_ptr + s)
    if exp > 0:
        if FUSED:
            ep2 = 2 * tl.load(ep_ptr) + PHASE
            tgt = ep2 * BIG + exp + 1
        else:
            tgt = tl.load(target_ptr + s)
        _ib.signal_wait_until(ctr_base_ptr + s, _CMP_EQ, tgt)          # NVSHMEM_CMP_EQ = 0


@triton.jit
def _k_recv_ids_from_hdr(recv_i32_ptr, dcnt_ptr, rrb_ptr, recv_ids_ptr,
                         WORLD: tl.constexpr, H: tl.constexpr, epg: tl.constexpr,
                         EPG_P: tl.constexpr, P_CAP: tl.constexpr, PCAP_P: tl.constexpr,
                         STRIDE: tl.constexpr, PZ_RRB: tl.constexpr = 0,
                         PZ_IDS: tl.constexpr = 0):
    """Rebuild the recv-side metadata from the HEADER ROWS that rode in with the data, instead of
    from the AllGathered histogram H. This is the edit that actually deletes the collective.

    grid (WORLD,): one program per source slot. Source s's header sits at row s*STRIDE + cnt_s,
    where cnt_s was decoded from the signal by `_wait_sig_slot` -- so the header is locatable without
    knowing anything another rank computed. Header int32 words (see `_k_write_header`):
        [0] cnt          [1] rev_recv_base (s's send_start for me)   [2 .. 2+epg) per-expert counts

    The bucket arithmetic below is COPIED from `_recv_ids_slotted_kernel` deliberately: only the
    SOURCE of `hsub` changes (header row vs H), so a divergence between the two isolates to the
    transport, not to the id derivation."""
    s = tl.program_id(0)
    cnt = tl.load(dcnt_ptr + s)
    w = H // 2                                                  # int32 words per bf16 row
    base = (s * STRIDE + cnt) * w
    # PZ_*: negative controls ONLY, off by default -- they corrupt one output each so the
    # corresponding oracle MUST fire, proving a clean run actually compared rather than skipped.
    tl.store(rrb_ptr + s, tl.load(recv_i32_ptr + base + 1).to(tl.int64) + PZ_RRB)
    ep = tl.arange(0, EPG_P)
    realm = ep < epg
    hsub = tl.load(recv_i32_ptr + base + 2 + ep, mask=realm, other=0).to(tl.int32)
    offs = tl.cumsum(hsub, axis=0)                              # inclusive, per-source
    offs = tl.where(realm, offs, P_CAP + 1)                     # padding never <= j
    pos = tl.arange(0, PCAP_P)
    valid = pos < cnt
    le = (offs[None, :] <= pos[:, None]).to(tl.int32)
    bucket = tl.sum(le, axis=1)
    eig_out = tl.where(valid & (bucket < epg), bucket, epg)      # header+padding -> sentinel epg
    if PZ_IDS:
        eig_out = tl.where(pos == 0, (eig_out + 1) % (epg + 1), eig_out)
    row = s * STRIDE + pos
    tl.store(recv_ids_ptr + row, eig_out.to(tl.int64), mask=(pos < STRIDE))


@triton.jit
def _wait_sig_slot(ctr_base_ptr, out_cnt_ptr, ep_ptr, WORLD: tl.constexpr,
                   PHASE: tl.constexpr, BIG: tl.constexpr, POISON: tl.constexpr = 0):
    """Slotted dispatch wait. Waits on ALL WORLD sources unconditionally (the MODE 0 put is
    unconditional, so every source signals every epoch) and DECODES the sender's row count out of
    the signal value -- which is what lets the receiver drop its dependence on the AllGathered
    histogram.

    CMP_GE, not CMP_EQ: the sender writes ep2*BIG + cnt + 1 and the receiver does not know `cnt`, so
    it cannot form an equality target. GE against ep2*BIG + 1 fires for any cnt >= 0. Epoch tagging
    still separates rounds -- a stale value from epoch ep2-1 is at most (ep2-1)*BIG + P_CAP + 1,
    which is below ep2*BIG + 1 because cnt+1 << BIG (P_CAP <= 720 vs BIG = 1e9). A value from a
    FUTURE epoch cannot have landed, since this very wait is what gates the sender's next round.

    Reading the signal object with a plain load after the wait is safe: the wait is the
    NVSHMEM-sanctioned synchronisation on that object, so the load is ordered after delivery and is
    not a conflicting access."""
    s = tl.program_id(0)
    ep2 = 2 * tl.load(ep_ptr) + PHASE
    base = ep2 * BIG
    _ib.signal_wait_until(ctr_base_ptr + s, _CMP_GE, base + 1)
    v = tl.load(ctr_base_ptr + s)
    # POISON: negative control ONLY. Off by default. Deliberately corrupts the decode so the
    # _DECODE_CHECK oracle must fire -- proving that a PASSING oracle run actually executed the
    # comparison rather than silently skipping it.
    tl.store(out_cnt_ptr + s, v - base - 1 + POISON)


@triton.jit
def _wait_poll(ctr_base_ptr, target_ptr, expected_ptr, ep_ptr, WORLD: tl.constexpr,
               PHASE: tl.constexpr, BIG: tl.constexpr, FUSED: tl.constexpr):
    # grid=(NBLK,): each block grid-strides over the sources it owns, spin-loading counter[s]==target[s].
    # NOT @requires_nvshmem: pure spin-load (a decorated no-nvshmem kernel fails cumodule_init err 7).
    # FUSED (L2): compute the poll target (2*ep+phase)*BIG+expected+1 IN-KERNEL from the already-loaded
    # `expected` -> removes the eager target.copy_ chain (~3 tiny long-arith kernels per _run). Pure
    # local arithmetic (no ordering risk); the `target` buffer is unused in the fused path.
    pid = tl.program_id(0)
    nblk = tl.num_programs(0)
    ep2 = 2 * tl.load(ep_ptr) + PHASE
    for s in tl.static_range(WORLD):
        if (s % nblk) == pid:
            exp = tl.load(expected_ptr + s)
            if exp > 0:
                if FUSED:
                    tgt = ep2 * BIG + exp + 1
                else:
                    tgt = tl.load(target_ptr + s)
                val = tl.load(ctr_base_ptr + s, volatile=True)
                while val != tgt:
                    val = tl.load(ctr_base_ptr + s, volatile=True)


class NcompassMoETransport:
    """Per-(comm, group) symmetric-buffer manager + device dispatch/combine.

    Buffers allocated once (collective rendezvous), sized to R_CAP/P_CAP rows (see
    ncompass_moe_config), falling back to a WORLD*max_send worst case when those are
    unset. One flags-zero + one barrier at
    allocation; the hot path has NO barrier (monotonic epoch handles reset).
    """

    _backend_set = False
    _logged_cfg = False

    def __init__(self, comm, H, dtype, dev, world, rank):
        self.comm, self.H, self.dtype, self.dev = comm, H, dtype, dev
        self.world, self.rank = world, rank
        self._esz = torch.empty((), dtype=dtype).element_size()   # data elt bytes (bf16=2) for IBGDA puts
        if _SLOTTED:
            _assert_slotted_preconditions()
            # P0 audit B5/B7/B8: these paths address the recv buffer DENSELY and would silently
            # disagree with slotted metadata. _PUTSIG is fine -- slots are built INTO it (F1).
            if not _PUTSIG:
                raise RuntimeError("NCOMPASS_G2_SLOTTED=1 requires NCOMPASS_G2_PUTSIG=1: slot "
                                   "addressing lives in _dispatch_put_sig; the legacy fence+flag "
                                   "put is dense and would corrupt silently.")
        if _PUTSIG and not _IBGDA:
            raise RuntimeError(
                "NCOMPASS_G2_PUTSIG=1 requires NCOMPASS_G2_IBGDA=1: put+signal is an IBGDA device\n"
                "primitive. Refusing to silently fall back to the legacy fence+flag transport."
            )
        if _IBGDA:
            _ib.install()                                          # set the wrapper+nvlink Triton hooks
        self.gname = (comm if comm is not None else dist.group.WORLD).group_name
        if not NcompassMoETransport._backend_set:
            # The torch symm_mem backend is a process-global that locks on first use. If a
            # sitecustomize hook (NCOMPASS_FORCE_NVSHMEM_BACKEND) or a prior instance already set
            # NVSHMEM, do NOT re-set: torch raises "Backend can not be changed after use" even for
            # the same name once any symm_mem op has run. Only set it if it isn't already NVSHMEM.
            try:
                cur = symm_mem.get_backend(dev)
            except Exception:
                cur = None
            if cur != "NVSHMEM":
                symm_mem.set_backend("NVSHMEM")
            NcompassMoETransport._backend_set = True
        self._bufs = {}
        # Monotonic epoch as a DEVICE tensor incremented IN the hot path, so it advances correctly on
        # CUDA-graph REPLAY (host code doesn't re-run on replay). Keeps each step's flags unique ->
        # no counter reset / barrier in the hot path. int64 headroom is ample (ep*BIG, BIG=1e9).
        self._ep = torch.zeros((), dtype=torch.int64, device=dev)
        # Worst-case CAP sizing (config-derived; launcher sets these). See PLAN_tier2 §10:
        #   R_CAP = dispatch-recv / combine-send rows = DP*max_num_batched_tokens*TOPK
        #           (all world pairs -> my group; the absolute pathological incoming bound).
        #   P_CAP = dispatch-send / combine-recv rows = (max_num_batched_tokens/SP)*TOPK
        #           (only my sequence-parallel shard's pairs come back).
        # If unset, fall back to the legacy single-cap heuristic (_cap).
        self.r_cap = _cfg.r_cap()
        self.p_cap = _cfg.p_cap()
        if _SLOTTED:
            # Checked HERE, not at the top of __init__: r_cap/p_cap are only assigned above.
            if self.p_cap is None or self.r_cap is None:
                raise RuntimeError(
                    "ncompass: the slotted layout requires resolved transport capacities, but "
                    "r_cap/p_cap are unset. ncompass_moe_config.resolve_caps() must run before "
                    "the transport is constructed (ncompass_moe_core calls it at the top of the "
                    "MoE forward); the fixed per-source slot is P_CAP rows wide and the layout is "
                    "undefined without it."
                )
            if self.r_cap != self.world * self.p_cap:
                raise RuntimeError(
                    f"NCOMPASS_G2_SLOTTED=1 requires R_CAP == WORLD*P_CAP (got r_cap={self.r_cap}, "
                    f"world={self.world}, p_cap={self.p_cap} -> {self.world * self.p_cap}). This "
                    "holds structurally when EP world == DP*SP, since R_CAP = DP*mnbt*TOPK and "
                    "P_CAP = (mnbt/SP)*TOPK; a mismatch means the slot layout would not tile the "
                    "recv buffer exactly."
                )
        # L2: fold the flag/target arithmetic into the transport kernels (removes the eager
        # fsrc/target .copy_ chains, ~0.5 ms/step of tiny long-arith kernels). Default off -> the
        # legacy precompute path stays bit-identical for rollback.
        # Default ON as of 2026-08-11. The eager chain below computes the epoch tag with ~18 int64
        # elementwise kernels per layer (22.13 us) that `_dispatch_put_sig` / `_wait_sig` already
        # compute in-kernel from the device epoch; measured -19.53 us/layer and +2.39% serve tok/s,
        # with BIT-IDENTICAL output on skew/hotspot/uniform/G_LIST=1,2,3. Set 0 to restore the eager
        # precompute path for rollback.
        self._fuse = _cfg.FUSE_FLAGS
        # L4: gather x[s_orig] straight into the send buffer via index_select(out=send) instead of
        # `x[s_orig].contiguous()` (temp alloc + vectorized_gather) followed by send.copy_ (a 2nd copy).
        # One kernel, no intermediate. Default off -> legacy gather+copy for rollback.

    def step(self):
        """Advance the epoch (call once per pipeline invocation). Device add_ -> captured, so it
        increments on every graph replay, keeping per-step flag values unique."""
        self._ep.add_(1)

    def debug_state_snapshot(self):
        """READ-ONLY dump of every piece of transport state that survives across a decode step.

        PROFILING INSTRUMENTATION ONLY (profiling-specs/2026-08-17-eplb-transport-rearrangement-state-
        mechanism.md, step E1). It mutates nothing, allocates only the host copies it returns, and is
        called from Python outside CUDA-graph capture. It exists because the EPLB x transport failure
        has had four mechanisms refuted and this state has never been observed at the boundary.

        What is dumped, and why each item is here:
          _ep    -- the device epoch. Every signal value is tagged `(2*ep+PHASE)*BIG + cnt + 1` and
                    `_wait_sig_slot` polls `>= (2*ep+PHASE)*BIG + 1`, so a cross-rank epoch
                    difference would make a sender's value unreachable (hang) or already-satisfied
                    (early read). Measured equal on 8 ranks at one rearrangement; measured here on
                    all 16, on both sides of the weight move.
          flags  -- the symmetric arrival counters peers write into. This is the object a stalled
                    receiver is spinning on.
          fsrc   -- the per-dest flag value a sender puts.
          target -- the per-source poll target (only populated on the non-FUSE path; kept because a
                    stale target is indistinguishable from a stale flag in a log without both).
          dcnt   -- per-source row counts decoded out of the signal value.
          arr    -- RELAY only: the Phase-R arrival flags, a second independent signal path.
        Values are compared ACROSS RANKS by the analysis; a single rank's dump proves nothing.
        """
        W = self.world
        out = {"rank": self.rank, "ep": int(self._ep.item()), "sets": {}}
        for key, b in self._bufs.items():
            e = {}
            for name in ("flags", "fsrc", "target", "dcnt", "hdr_rrb", "arr"):
                t = b.get(name)
                if t is None:
                    continue
                e[name] = t[:W].detach().to("cpu", copy=True).tolist()
            scr = b.get("scr")
            if scr is not None:
                # scr[0] is ALSO the signal sink for inactive dests (`sg = tl.where(active, sg,
                # scr_ptr)` in _dispatch_put_sig MODE 2), so the low words hold clobbered signal
                # values, not diagnostics. L1's epoch-range counters deliberately live at [8] and [9]
                # to stay clear of it: [8] = worst out-of-range decoded count, [9] = how many times.
                e["scr"] = scr[:12].detach().to("cpu", copy=True).tolist()
            out["sets"][str(key)] = e
        return out

    def _grp(self, key, send_cap, recv_cap):
        # key = (group, phase): dispatch and combine MUST use separate recv buffers, else combine's
        # puts to a peer's recv clobber that peer's dispatch-recv (Marlin's input) with no cross-rank
        # sync -> garbage. Separate buffer sets per (g, phase) eliminate the interaction.
        # send/recv sized independently to worst-case: dispatch {send=P_CAP, recv=R_CAP},
        # combine {send=R_CAP, recv=P_CAP}. All-symmetric, no reuse (deferred opts: PLAN_tier2 §9).
        b = self._bufs.get(key)
        if b is None:
            H, dt, dev, W = self.H, self.dtype, self.dev, self.world
            # One-line rank-0 config echo. The serve launchers set these gates by env and a missing
            # `-e` line silently runs a DIFFERENT path while every health check and benchmark still
            # passes (the M5 class of bug; one result was retracted for it). An A/B is only
            # interpretable if the arm it claims to be is OBSERVED in the engine log, not inferred.
            if self.rank == 0 and not NcompassMoETransport._logged_cfg:
                NcompassMoETransport._logged_cfg = True
                print(f"[ncompass transport] RELAY={int(_RELAY)} DWARPS={_DWARPS} RWARPS={_RWARPS} "
                      f"RBLK={_RBLK} SLOTTED={int(_SLOTTED)} NOAG={int(_NOAG)} B2B={int(_B2B)} "
                      f"PUTSIG={int(_PUTSIG)} IBGDA={int(_IBGDA)}", flush=True)
            b = {
                "send": symm_mem.empty(send_cap, H, dtype=dt, device=dev),
                "recv": symm_mem.empty(recv_cap, H, dtype=dt, device=dev),
                "flags": symm_mem.empty(max(W, 64), dtype=torch.int64, device=dev),
                "fsrc": symm_mem.empty(W, dtype=torch.int64, device=dev),
                "target": torch.zeros(max(W, 64), dtype=torch.int64, device=dev),
                # slotted: per-source row counts decoded from the signal value (local, not AllGathered)
                "dcnt": torch.zeros(max(W, 64), dtype=torch.int64, device=dev),
                # slotted: recv metadata rebuilt from the header rows (replaces H entirely)
                "hdr_rrb": torch.zeros(max(W, 64), dtype=torch.int64, device=dev),
                "hdr_ids": torch.zeros(recv_cap, 1, dtype=torch.int64, device=dev),
                "send_cap": send_cap, "recv_cap": recv_cap,
            }
            if _RELAY:
                # rail/land: the Phase-R payload [NTOK tokens][desc][meta][W hdr rows].
                # stage: Phase N's contiguous per-dest staging. GPN*STRIDE rows is the dominant
                # cost here (~82 MB at P_CAP=720) and it exists only because Phase N forwards with
                # putmem_signal_block; the S1 scatter variant needs no staging but does need the
                # visibility protocol, which is S3.
                _st = SLOT_STRIDE(self.p_cap or 1)
                b["rail"] = symm_mem.empty(send_cap + 2 + W, H, dtype=dt, device=dev)
                b["land"] = symm_mem.empty(send_cap + 2 + W, H, dtype=dt, device=dev)
                b["arr"] = symm_mem.empty(max(W, 64), dtype=torch.int64, device=dev)
                b["scr"] = symm_mem.empty(64, dtype=torch.int64, device=dev)
                b["stage"] = symm_mem.empty((W // 2) * _st, H, dtype=dt, device=dev)
                for k in ("rail", "land", "arr", "scr", "stage"):
                    symm_mem.rendezvous(b[k], self.gname)
                b["arr"].zero_(); b["scr"].zero_()
            for k in ("send", "recv", "flags", "fsrc"):
                symm_mem.rendezvous(b[k], self.gname)
            b["flags"].zero_()
            dist.barrier(group=self.comm)      # one-time: all ranks zeroed before any flag lands
            self._bufs[key] = b
        return b

    def _cap(self, need):  # legacy fallback when R_CAP/P_CAP unset
        return max(need, self.world * max(need, 1))

    def _caps_for(self, phase, P, R):
        # (send_cap, recv_cap). dispatch(0): I send P rows, receive R rows.
        # combine(1): I send R rows (my expert outputs), receive P rows (results home).
        if self.r_cap is not None and self.p_cap is not None:
            if phase == 0:
                # SLOTTED (L1): the dispatch send buffer gains one header row per destination so each
                # dest's [data..][header] region is contiguous for a single put.
                if _SLOTTED:
                    # send: +WORLD header rows (one per dest, contiguous with that dest's data).
                    # recv: WORLD slots of (P_CAP+1). See SLOT_GEOMETRY above -- this is F2 of the P0
                    # audit; the old WORLD*P_CAP sizing under-allocates by exactly WORLD rows.
                    return (self.p_cap + self.world, self.world * SLOT_STRIDE(self.p_cap))
                return (self.p_cap, self.r_cap)
            # combine: send = Marlin's output, which must have EXACTLY as many rows as its input
            # (the dispatch recv buffer). Under slots that is WORLD*(P_CAP+1), not r_cap -- otherwise
            # moe_wna16_marlin_gemm raises "Shape mismatch: c.size(0)".
            if _SLOTTED:
                return (self.world * SLOT_STRIDE(self.p_cap), self.p_cap)
            return (self.r_cap, self.p_cap)
        c = self._cap(max(P, R))
        return c, c

    def _run(self, b, cnt, sstart, rbase, expected, phase, src_tensor, pre_staged=False,
             gather_idx=None, my_hist=None, grp=0, epg=0, e_local=0):
        """One device a2a: copy src into the (static) send buffer, put per-dest chunks by DEVICE counts,
        poll. src_tensor.shape[0] is a static shape (not a host sync) -> capturable. recv lands in b.

        pre_staged: src_tensor IS already the send buffer (producer wrote into it directly, e.g. Marlin's
        2nd-GEMM output fused into the combine-send buffer) -> skip the CAP-sized send-copy entirely.
        gather_idx (L4): src_tensor is the FULL x ([T+1,H]); gather x[gather_idx] straight into the send
        buffer via index_select(out=) -> no intermediate gather tensor + no separate send-copy."""
        W, H = self.world, self.H
        send, recv, flags, fsrc, target = b["send"], b["recv"], b["flags"], b["fsrc"], b["target"]
        if _SLOTTED and phase == 0 and not pre_staged:
            # L1 dispatch fill: gather with the +d shift so each dest's group is followed by its own
            # header row, then write the headers. Replaces the flat index_select.
            # BOTH dispatch variants must land here. `_fuse_gather` decides only whether the source is
            # the full x (gather_idx given) or an already-gathered dense buffer; either way the rows
            # need the +d shift, because the put unconditionally reads dest d's block at sstart[d]+d.
            # Previously this branch also required `gather_idx is not None`, so the non-fused path fell
            # through to a DENSE copy while the put stayed shifted -- every receiver then read its own
            # rank's worth of rows too late (measured delta == ep_rank, rank 0 exact since its shift is 0).
            _idx = gather_idx is not None
            n = gather_idx.shape[0] if _idx else src_tensor.shape[0]
            assert n + W <= send.shape[0], \
                f"transport slotted send OOB: {n}+{W} > send_cap {send.shape[0]} -- raise P_CAP"
            assert my_hist is not None and epg > 0, \
                "slotted dispatch needs my_hist/epg for the header row (pass them from dispatch())"
            if _B2B:
                # B2b-2: headers ride the gather's tail. Grid is max(n, W) so the W header programs
                # always exist even when n < W (small batch) -- otherwise some destination would
                # silently ship no header and its peer would decode garbage.
                _k_gather_slotted[(max(n, W),)](
                    src_tensor, send, gather_idx if _idx else send, sstart, cnt,
                    send.view(torch.int32), my_hist, self._ep, n,
                    WORLD=W, H=H, BLK_H=1024, SHIFT=1, IDX=(1 if _idx else 0), HDR=1,
                    EPG_P=triton.next_power_of_2(epg), epg=epg, E_LOCAL=e_local, GRP=grp,
                    PHASE=phase, BIG=_BIG,
                    RELAY=(1 if (_RELAY and phase == 0) else 0), GPN=W // 2,
                    MYNODE=self.rank // (W // 2),
                    rail_ptr=(b["rail"] if _RELAY else None),
                    rail_i32_ptr=(b["rail"].view(torch.int32) if _RELAY else None),
                    NTOK=(src_tensor.shape[0] if _RELAY else 0), num_warps=8)
            else:
                _k_gather_slotted[(n,)](src_tensor, send,
                                        gather_idx if _idx else send,  # unused when IDX=0
                                        sstart, cnt, send, send, send, n,
                                        WORLD=W, H=H, BLK_H=1024,
                                        SHIFT=1, IDX=(1 if _idx else 0), HDR=0, num_warps=8)
                _k_write_header[(W,)](send.view(torch.int32), cnt, sstart, my_hist, self._ep,
                                      WORLD=W, H=H, EPG_P=triton.next_power_of_2(epg), epg=epg,
                                      E_LOCAL=e_local, GRP=grp, PHASE=phase, BIG=_BIG, num_warps=1)
        elif not pre_staged:
            if gather_idx is not None:
                n = gather_idx.shape[0]
                assert n <= send.shape[0], \
                    f"transport send OOB: gather {n} > send_cap {send.shape[0]} -- raise R_CAP/P_CAP"
                torch.index_select(src_tensor, 0, gather_idx, out=send[:n])  # gather -> send, one kernel
            else:
                assert src_tensor.shape[0] <= send.shape[0], \
                    f"transport send OOB: src {src_tensor.shape[0]} > send_cap {send.shape[0]} -- raise R_CAP/P_CAP"
                send[:src_tensor.shape[0]].copy_(src_tensor)  # static slice; put reads only valid (cnt) rows
        if not self._fuse:
            ep2 = 2 * self._ep + phase
            fsrc.copy_(ep2 * _BIG + cnt + 1)              # flag value per dest (device cnt)
            target[:W].copy_(ep2 * _BIG + expected + 1)   # poll target per source (device expected)
        # FUSED path: _dispatch_put computes fsrc[d] and _wait_poll computes its target in-kernel from
        # self._ep (device epoch) + phase -> the two eager .copy_ chains above are skipped entirely.
        nblk = W
        if _RELAY and phase == 0 and _SLOTTED and _PUTSIG and _IBGDA:
            # S2 two-hop, SERIALISED (no chunking): Phase R then Phase N then the unchanged wait.
            # Combine (phase 1) stays on the direct path -- the scope bound in spec rev 10 section 7.
            GPN, NTOK = W // 2, src_tensor.shape[0]
            _ST = SLOT_STRIDE(self.p_cap or 1)
            _dispatch_put_sig[(W,)](
                recv, send, cnt, sstart, rbase, flags, self.rank, fsrc, self._ep,
                WORLD=W, H=H, ESZ=self._esz, PHASE=phase, BIG=_BIG, FUSED=self._fuse,
                SLOT=1, STRIDE=_ST, MODE=2,
                land_ptr=b["land"], rail_ptr=b["rail"], arr_ptr=b["arr"], scr_ptr=b["scr"],
                GPN=GPN, NTOK=NTOK, RGUARD=0, RDIAG=0, num_warps=_DWARPS)
            _relay_fanout[(GPN,)](
                recv, b["land"], b["land"].view(torch.int32), b["stage"], flags, b["arr"],
                self._ep, b["scr"], self.rank, WORLD=W, GPN=GPN, H=H, ESZ=self._esz,
                STRIDE=_ST, NTOK=NTOK, PHASE=phase, BIG=_BIG, SCAP=b["send_cap"],
                BLK_H=_RBLK, NOWAIT=0, num_warps=_RWARPS)
            _wait_sig_slot[(W,)](flags, b["dcnt"], self._ep, WORLD=W, PHASE=phase, BIG=_BIG,
                                 POISON=0)
        elif _PUTSIG and _IBGDA:
            # L2 rev 6: fused put+signal, DENSE layout, BOTH legs. Drops the fence, the separate flag
            # put and the SM spin.
            _dispatch_put_sig[(W,)](
                recv, send, cnt, sstart, rbase, flags, self.rank, fsrc, self._ep,
                WORLD=W, H=H, ESZ=self._esz, PHASE=phase, BIG=_BIG, FUSED=self._fuse,
                SLOT=(1 if _SLOTTED else 0), STRIDE=SLOT_STRIDE(self.p_cap or 1),
                MODE=(0 if phase == 0 else 1), num_warps=_DWARPS)
            if _SLOTTED and phase == 0:
                # decodes recv counts into b["dcnt"]; P1b-2 makes this the ONLY source of them.
                _wait_sig_slot[(W,)](flags, b["dcnt"], self._ep, WORLD=W, PHASE=phase, BIG=_BIG,
                                     POISON=0)
            else:
                _wait_sig[(W,)](flags, target, expected, self._ep,
                                WORLD=W, PHASE=phase, BIG=_BIG, FUSED=self._fuse)
        elif _SLOTTED:
            # L1: slot-addressed put. Signalling stays fence+flag (P2 changes only that), so this
            # phase validates the LAYOUT in isolation and a rel_l2 failure attributes to the layout.
            _dispatch_put_slotted[(nblk,)](
                recv, send, cnt, sstart, rbase, flags, self.rank, fsrc, self._ep,
                WORLD=W, H=H, ESZ=self._esz, PHASE=phase, BIG=_BIG,
                P_CAP=self.p_cap, MODE=(0 if phase == 0 else 1))
            _wait_poll[(nblk,)](flags, target, expected, self._ep,
                                WORLD=W, PHASE=phase, BIG=_BIG, FUSED=self._fuse)
        else:
            if _IBGDA:
                _dispatch_put_ib[(nblk,)](recv, send, cnt, sstart, rbase, flags, self.rank, fsrc, self._ep,
                                          WORLD=W, H=H, ESZ=self._esz, PHASE=phase, BIG=_BIG, FUSED=self._fuse)
            else:
                _dispatch_put[(nblk,)](recv, send, cnt, sstart, rbase, flags, self.rank, fsrc, self._ep,
                                       WORLD=W, H=H, PHASE=phase, BIG=_BIG, FUSED=self._fuse)
            _wait_poll[(nblk,)](flags, target, expected, self._ep,
                                WORLD=W, PHASE=phase, BIG=_BIG, FUSED=self._fuse)
    def _relay_check(self, b, cnt, sstart, rbase, src_tensor, gather_idx,
                     my_hist, grp, epg, e_local, phase):
        """S2 Phase-R oracle (NCOMPASS_G2_RELAY_CHECK=1, debug only -- host syncs + a collective).

        Runs the RELAY gather and the MODE=2 put into SHADOW buffers, so the live dispatch is
        untouched and its output stays the ground truth. Same pattern as _DECODE_CHECK: verify
        against a truth that is still live, before anything depends on the new path.

        Four checks. C3 is the one that matters most -- a mis-selected signal address would set a
        remote source's flag BEFORE the relay wrote its rows, and the receiver would then read stale
        memory and carry on. That failure is silent end-to-end."""
        W, H = self.world, self.H
        GPN = W // 2
        mynode, dev, dt = self.rank // GPN, self.dev, self.dtype
        NTOK = src_tensor.shape[0]
        STRIDE = SLOT_STRIDE(self.p_cap or 1)
        n = gather_idx.shape[0]
        s = self._bufs.get("relaychk")
        if s is None:
            cap = b["send_cap"]
            s = {"send": symm_mem.empty(cap, H, dtype=dt, device=dev),
                 "recv": symm_mem.empty(b["recv_cap"], H, dtype=dt, device=dev),
                 "flags": symm_mem.empty(max(W, 64), dtype=torch.int64, device=dev),
                 "fsrc": symm_mem.empty(W, dtype=torch.int64, device=dev),
                 "rail": symm_mem.empty(cap + 2 + W, H, dtype=dt, device=dev),
                 "land": symm_mem.empty(cap + 2 + W, H, dtype=dt, device=dev),
                 "arr": symm_mem.empty(max(W, 64), dtype=torch.int64, device=dev),
                 "scr": symm_mem.empty(64, dtype=torch.int64, device=dev)}
            for k in ("send", "recv", "flags", "fsrc", "rail", "land", "arr", "scr"):
                symm_mem.rendezvous(s[k], self.gname)
            dist.barrier(group=self.comm)
            self._bufs["relaychk"] = s
        s["flags"].zero_(); s["arr"].zero_(); s["scr"].zero_(); s["land"].zero_()
        dist.barrier(group=self.comm)

        _k_gather_slotted[(max(n, W),)](
            src_tensor, s["send"], gather_idx, sstart, cnt,
            s["send"].view(torch.int32), my_hist, self._ep, n,
            WORLD=W, H=H, BLK_H=1024, SHIFT=1, IDX=1, HDR=1,
            EPG_P=triton.next_power_of_2(epg), epg=epg, E_LOCAL=e_local, GRP=grp,
            PHASE=phase, BIG=_BIG, RELAY=1, GPN=GPN, MYNODE=mynode,
            rail_ptr=s["rail"], rail_i32_ptr=s["rail"].view(torch.int32),
            NTOK=NTOK, num_warps=8)
        torch.cuda.synchronize(); dist.barrier(group=self.comm)
        _dispatch_put_sig[(W,)](
            s["recv"], s["send"], cnt, sstart, rbase, s["flags"], self.rank, s["fsrc"], self._ep,
            WORLD=W, H=H, ESZ=self._esz, PHASE=phase, BIG=_BIG, FUSED=self._fuse,
            SLOT=1, STRIDE=STRIDE, MODE=2,
            land_ptr=s["land"], rail_ptr=s["rail"], arr_ptr=s["arr"], scr_ptr=s["scr"],
            GPN=GPN, NTOK=NTOK)
        torch.cuda.synchronize(); dist.barrier(group=self.comm)

        bad = []
        # C3 -- signal routing. Only intra-node sources may have signalled MY flags; the 8 remote
        # sources must NOT have, because their rows come via their relay.
        fl = s["flags"][:W].tolist()
        for src in range(W):
            intra = (src // GPN) == mynode
            if intra and fl[src] == 0:
                bad.append(f"C3 intra src {src} did not signal")
            if (not intra) and fl[src] != 0:
                bad.append(f"C3 REMOTE src {src} signalled early (val={fl[src]})")
        # C3b -- my rail partner's Phase R must have signalled my ARRIVAL counter, and only that one
        partner = self.rank ^ GPN
        av = s["arr"][:W].tolist()
        if av[partner] == 0:
            bad.append(f"C3b partner {partner} did not signal arr")
        if any(v != 0 for i, v in enumerate(av) if i != partner):
            bad.append(f"C3b arr signalled by a non-partner: {av}")
        # C1 -- my landing rows must be my partner's token block, byte-exact
        allx = torch.empty(W * NTOK, H, dtype=dt, device=dev)
        dist.all_gather_into_tensor(allx, src_tensor[:NTOK].contiguous(), group=self.comm)
        want_x = allx[partner * NTOK:(partner + 1) * NTOK]
        if not torch.equal(s["land"][:NTOK], want_x):
            nb = (s["land"][:NTOK] != want_x).any(dim=1).sum().item()
            bad.append(f"C1 landing != partner x: {nb}/{NTOK} rows differ")
        # C2 -- descriptors, now read out of the LANDING payload the partner actually shipped, not
        # out of my own rail buffer: this checks the wire, not just the gather.
        rw = H // 2
        li = s["land"].view(torch.int32).reshape(-1)   # 2-D view is [rows, H/2]; flatten for element slicing
        pv = torch.arange(n, device=dev).view(n, 1)
        allsc = torch.empty(W * W, dtype=torch.int64, device=dev)
        allss = torch.empty(W * W, dtype=torch.int64, device=dev)
        dist.all_gather_into_tensor(allsc, cnt[:W].contiguous(), group=self.comm)
        dist.all_gather_into_tensor(allss, sstart[:W].contiguous(), group=self.comm)
        pc, ps = allsc[partner * W:(partner + 1) * W], allss[partner * W:(partner + 1) * W]
        pnode = partner // GPN
        dv = (pv >= (ps.view(1, W) + pc.view(1, W))).sum(dim=1)
        allso = torch.empty(W * n, dtype=torch.int64, device=dev)
        dist.all_gather_into_tensor(allso, gather_idx[:n].contiguous(), group=self.comm)
        pso = allso[partner * n:(partner + 1) * n].to(torch.int32)
        want_d = torch.where((dv // GPN) != pnode, pso, torch.full_like(pso, -1))
        got_d = li[NTOK * rw:NTOK * rw + n]
        if not torch.equal(got_d, want_d):
            bad.append(f"C2 desc differs on {(got_d != want_d).sum().item()}/{n} pairs")
        # C5 -- meta row: the partner's cnt/sstart as the relay will read them
        if not (torch.equal(li[(NTOK + 1) * rw:(NTOK + 1) * rw + W], pc.to(torch.int32))
                and torch.equal(li[(NTOK + 1) * rw + W:(NTOK + 1) * rw + 2 * W],
                                ps.to(torch.int32))):
            bad.append("C5 meta row (cnt/sstart) mismatch")
        # C6 -- forwarded header rows: cnt must agree with the meta row for every local dest
        for jj in range(GPN):
            dd = mynode * GPN + jj
            hb = (NTOK + 2 + jj) * rw
            if int(li[hb]) != int(pc[dd]):
                bad.append(f"C6 hdr row {jj} cnt={int(li[hb])} != meta cnt[{dd}]={int(pc[dd])}")
        # C4 -- MODE=2 must not perturb the intra path it inherits from MODE=0
        for src in range(W):
            if (src // GPN) == mynode and not torch.equal(
                    s["recv"][src * STRIDE:(src + 1) * STRIDE],
                    b["recv"][src * STRIDE:(src + 1) * STRIDE]):
                bad.append(f"C4 intra slot {src} differs from the live dispatch")
        tag = f"[relay-check rank {self.rank} NTOK={NTOK} n={n}]"
        if bad:
            raise RuntimeError(f"{tag} FAIL: " + "; ".join(bad[:8]))
        print(f"{tag} PASS C1 C2 C3 C3b C4", flush=True)

    def dispatch(self, g, m, x):
        """Route x[s_orig] to expert-owner ranks. Returns recv_x[R,H] (source-ordered)."""
        P, R = m["P"], m["R"]
        b = self._grp((g, 0), *self._caps_for(0, P, R))  # dispatch: send=P_CAP, recv=R_CAP
        sc, ss, rb, rc = (m["send_counts_dev"], m["send_start_dev"],
                          m["recv_base_dev"], m["recv_counts_dev"])
        # x is x_ext ([T+1,H]); s_orig padding rows = T -> gather the zero dump row. Full static gather.
        if True:   # slotted gather is unconditional
            # L4: index_select x[s_orig] directly into the send buffer (no temp, no separate copy).
            # The C2/C1 fill MUST take this path too: `_k_gather_slotted` already performs the
            # x[s_orig] gather itself, so pre-gathering into `src` first would do the full
            # [MAXP, H] gather TWICE per iteration (measured: a redundant 2.99 us kernel plus
            # ~20-30 us of launch-thread time, which dominates in this launch-bound pipeline).
            self._run(b, sc, ss, rb, rc, phase=0, src_tensor=x, gather_idx=m["s_orig"],
                      my_hist=m.get("my_hist_dev"), grp=m.get("grp", 0),
                      epg=m.get("epg", 0), e_local=m.get("e_local", 0))
        else:
            src = x[m["s_orig"]].contiguous()
            self._run(b, sc, ss, rb, rc, phase=0, src_tensor=src)
        if _SLOTTED:
            # L3 P1b-2: rebuild the recv-side metadata from the header rows that arrived WITH the
            # data. Runs after _run (which contains the wait), so the headers are guaranteed landed.
            # While the AllGather is still live this is checked against it; under _NOAG it BECOMES
            # the metadata and the collective is gone.
            epg_m, p_cap = m.get("epg", 0), self.p_cap
            # hdr_ids is zero-initialised and 0 is a VALID expert id, so the kernel must cover EVERY
            # row -- any uncovered tail would silently read as expert 0 instead of the epg sentinel.
            # _caps_for makes this exact by construction; assert it rather than assume it.
            assert b["hdr_ids"].shape[0] == self.world * SLOT_STRIDE(p_cap), (
                f'slotted recv cap {b["hdr_ids"].shape[0]} != WORLD*STRIDE '
                f'{self.world * SLOT_STRIDE(p_cap)} -- header ids would leave rows as expert 0')
            _k_recv_ids_from_hdr[(self.world,)](
                b["recv"].view(torch.int32), b["dcnt"], b["hdr_rrb"], b["hdr_ids"],
                WORLD=self.world, H=self.H, epg=epg_m,
                EPG_P=triton.next_power_of_2(epg_m), P_CAP=p_cap,
                PCAP_P=triton.next_power_of_2(p_cap), STRIDE=SLOT_STRIDE(p_cap),
                PZ_RRB=0,
                PZ_IDS=0)
            if _NOAG:
                m["recv_counts_dev"] = b["dcnt"][:self.world]
                m["rev_recv_base_dev"] = b["hdr_rrb"][:self.world]
                m["recv_topk_ids"] = b["hdr_ids"]
        # CAP mode: return the whole recv buffer (matches the CAP recv_topk_ids; padding rows [R:] are
        # masked in Marlin via sentinel expert ids). Legacy mode: tight recv[:R].
        return b["recv"] if self.r_cap is not None else b["recv"][:R]

    def combine_send_buf(self, g, m):
        """The symmetric send buffer for group g's combine phase ([R_CAP, H]). Cap mode only.
        Producers (Marlin) can write their output straight into this to skip the CAP-sized send-copy;
        combine() then auto-detects the aliasing (data_ptr) and runs pre-staged."""
        if self.r_cap is None:
            return None
        return self._grp((g, 1), *self._caps_for(1, m["P"], m["R"]))["send"]

    def combine(self, g, m, y):
        """Reverse a2a: expert outputs y[R,H] back to source ranks. Returns y_back[P,H]
        (dest-ordered, same as the original send -> caller scatter-adds via s_orig)."""
        P, R = m["P"], m["R"]
        b = self._grp((g, 1), *self._caps_for(1, P, R))  # combine: send=R_CAP, recv=P_CAP
        rc, sc = m["recv_counts_dev"], m["send_counts_dev"]
        rev_ss = (torch.cumsum(rc, 0) - rc).contiguous()   # my send row per source
        # SLOTTED: my recv buffer is WORLD fixed slots of STRIDE=P_CAP+1 rows, LEFT-aligned, so
        # source s's data starts at s*STRIDE -- unrelated to the dense cumsum. The put kernel
        # computes that locally (MODE=1 uses so = s*STRIDE), so `rev_ss` is unused under slots.
        rev_rb = m["rev_recv_base_dev"]                     # where my rows land in each source's y_back
        # y is CAP-sized (Marlin ran on the CAP recv buffer); the put sends only the R valid rows via
        # device counts. If y IS the send buffer (Marlin fused its output in), skip the copy (pre_staged).
        # data_ptr is host-side -> resolved at capture, so the branch is baked into the graph.
        pre = (self.r_cap is not None and y.numel() > 0
               and y.data_ptr() == b["send"].data_ptr())
        self._run(b, rc, rev_ss, rev_rb, sc, phase=1,
                  src_tensor=(y if pre else y.contiguous()), pre_staged=pre)
        return b["recv"] if self.r_cap is not None else b["recv"][:P]
