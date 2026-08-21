# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ncompass: the shipped MoE decode configuration, as constants.

SINGLE SOURCE OF TRUTH. These values used to be independent `os.environ` reads scattered
across `ncompass_transport`, `ncompass_routing_meta` and `ncompass_moe_core` -- and two of those
modules read the SAME names with their own defaults, so a value could diverge between the
transport and the metadata that addresses it. That is not a hypothetical: the whole set
defaulted to OFF in code while every measured winning run set them ON from a shell script,
which meant an unset environment produced a healthy engine quietly serving the slow path.

The values below are the configuration measured at 8368.96 tok/s (n=8, 4 boots) on
Kimi-K2.6 decode, 2x8 H100, TP4/DP4/DCP4/EP16, W4A16 Marlin, FULL_DECODE_ONLY, EPLB on.
They are recovered from the live engine's own startup lines, which are also the positive
assertion that the optimized path is running (see docs/ncompass-kimi-decode-moe.md section 7):

    [ncompass transport] RELAY=1 DWARPS=8 RWARPS=32 RBLK=8192 SLOTTED=1 NOAG=1 B2B=1 PUTSIG=1 IBGDA=1
    ncompass MoE: WORLD=16 E_LOCAL=24 epg=24 linear_placement=True
    ncompass caps: P_CAP=720 R_CAP=11520

These are implementation details of one validated configuration, not knobs. Do not
reintroduce environment gates here. If a path is worth keeping, make it unconditional; if
it is not, delete it.

Reaching this module at all requires `VLLM_NCOMPASS_KIMI_DECODE_MOE=1`; see
`compressed_tensors_moe_wna16_marlin.py`. The flag governs CODE only -- the deployment
still has to supply the IBGDA wrapper, the NVSHMEM symm-mem backend, and EPLB's serve
args. See docs/ncompass-kimi-decode-moe.md.
"""


# --- transport ---------------------------------------------------------------------
# Recv buffer is WORLD fixed slots of P_CAP rows, source s owning slot s in every
# destination, written right-aligned so one contiguous put always ends at a fixed address.
SLOTTED = True
# Rail relay: cross each remote payload the NIC once to paired=rank^8, which NVLink-forwards.
RELAY = True
# Fused put+signal: one network op per peer carrying data AND its completion signal.
PUTSIG = True
# IBGDA device-side wrappers. Built in-image by ncompass_ibgda/build_wrapper.sh; nothing is mounted.
IBGDA = True
# Metadata AllGather deleted; every recv-side quantity is recovered from the transport.
NOAG = True

# Relay kernel geometry (measured).
RELAY_WARPS = 32
RELAY_BLOCK = 8192
# Dispatch warps. Was `8 if RELAY else 4`; RELAY is True, so 8.
DISPATCH_WARPS = 8

# --- metadata ----------------------------------------------------------------------
# B2b: skip metadata the transport already supplies; fold the header into the gather.
B2B = True
# B2a: the whole combine epilogue in one Triton kernel.
FUSED_EPI = True
# Fuse the transport's flag writes.
FUSE_FLAGS = True

# --- compute -----------------------------------------------------------------------
# Expert groups. G=1 is shipped. The pipeline is written group-generically and the G=2
# variant was measured as a net regression at decode; the generality is retained only
# because removing it is a structural refactor with no measured payoff.
EXPERT_GROUPS = 1


# --- transport capacities -----------------------------------------------------------
# R_CAP / P_CAP size the fixed all-to-all slots. They were read from the environment in
# THREE independent places (the transport's __init__, ncompass_moe_core, ncompass_routing_meta),
# each with its own default, and the values themselves were computed in a launcher shell
# script -- so the engine could not be started without that script.
#
# Derived here instead:
#
#     P_CAP = ceil(max_num_batched_tokens / TP) * topk
#     R_CAP = EP_WORLD * P_CAP
#
# The MoE is sequence-parallel sharded across TP, so each rank sees ceil(max_batch/TP)
# distinct tokens, and P_CAP bounds the send rows one rank can produce for one destination.
#
# R_CAP IS DERIVED FROM P_CAP, NOT INDEPENDENTLY. The launcher used
# `DP * max_batch * topk`, which equals `EP_WORLD * P_CAP` only when TP divides
# max_num_batched_tokens (it does at the shipped point: ceil(360/4)*8*16 = 11520 =
# 4*360*8). When it does not divide, the two disagree and the slotted layout's
# `R_CAP == WORLD * P_CAP` invariant fails at transport construction. Deriving R_CAP from
# P_CAP makes that invariant true by construction instead of by arithmetic coincidence.
#
# CAP ARITHMETIC IS SHARP. `x_ext` is [T+1, H], so a batch at exactly T*topk == P_CAP still
# overflows by one row and faults with an illegal memory access in `_relay_fanout`. Do not
# "tidy" these formulas without re-running the transport.
_CAPS: tuple[int, int] | None = None
_CAPS_TOPK: int | None = None




_SNAPSHOT: dict[str, int] | None = None
_VALIDATED = False


def snapshot_config() -> None:
    """Capture the config-derived scalars while a vLLM config context is live, and validate.

    MUST be called from model-construction time (the MoE method's `__init__`). It cannot be
    done lazily from the forward pass: `get_current_vllm_config()` reads a context variable
    that `set_current_vllm_config()` populates around model construction only, so calling it
    from a forward raises "Current vLLM config is not set". That is not a theoretical
    concern -- an earlier revision of this file did exactly that and every worker died at
    the first MoE forward, before the caps were ever computed.

    Validating here too means a misconfigured engine fails during model init with a message
    naming the problem, instead of surviving to the first decode step.
    """
    global _SNAPSHOT
    if _SNAPSHOT is not None:
        return
    from vllm.config import get_current_vllm_config

    cfg = get_current_vllm_config()
    _SNAPSHOT = {
        "max_batch": int(cfg.scheduler_config.max_num_batched_tokens),
        "tp": int(cfg.parallel_config.tensor_parallel_size),
        "dp": int(cfg.parallel_config.data_parallel_size),
    }
    validate_environment()




def validate_environment() -> None:
    """Refuse to start when the flag is on but the engine is configured incompatibly.

    Every one of these was, at some point, learned the hard way and at a distance from its
    cause. Left unchecked they do not produce a configuration error -- they produce an
    illegal memory access during CUDA-graph capture, or a transport that silently cannot
    claim the symmetric-memory backend, both of which surface as something else entirely.

    This validates rather than overrides. If an operator explicitly asked for
    `fuse_allreduce_rms=true`, quietly ignoring them would be worse than refusing: they
    would get a working engine that is not the one they configured.
    """
    global _VALIDATED
    if _VALIDATED:
        return
    _VALIDATED = True

    import vllm.envs as envs
    from vllm.config import get_current_vllm_config

    cfg = get_current_vllm_config()   # caller guarantees a live config context
    problems: list[str] = []

    if envs.VLLM_ALLREDUCE_USE_SYMM_MEM:
        problems.append(
            "VLLM_ALLREDUCE_USE_SYMM_MEM must be 0. vLLM's symm-mem all-reduce claims the "
            "process-global torch symmetric-memory backend with the CUDA backend before the "
            "MoE runs; the backend locks on first use, so the transport can then never set "
            "NVSHMEM and dies later with an unrelated-looking error."
        )

    comp = cfg.compilation_config
    if getattr(comp.pass_config, "fuse_allreduce_rms", False):
        problems.append(
            "-cc.pass_config.fuse_allreduce_rms must be false. The fusion collides with the "
            "NVSHMEM symmetric allocations and kills FULL_DECODE_ONLY capture with a "
            "deterministic illegal memory access -- reproducibly at layer 152, with nothing "
            "in the traceback pointing at this pass."
        )

    if not cfg.parallel_config.disable_custom_all_reduce:
        problems.append(
            "--disable-custom-all-reduce is required. CustomAllreduce's one-shot kernel takes "
            "an illegal memory access during capture alongside the NVSHMEM allocations."
        )

    if problems:
        raise RuntimeError(
            "VLLM_NCOMPASS_KIMI_DECODE_MOE=1 is set, but this engine is configured in a way "
            "the optimized decode path cannot run under:\n\n  - "
            + "\n  - ".join(problems)
            + "\n\nSee docs/ncompass-kimi-decode-moe.md for the "
              "full deployment contract, or unset the flag to run stock vLLM."
        )


def resolve_caps(topk: int, ep_world: int) -> tuple[int, int]:
    """Resolve (R_CAP, P_CAP) once, from the running vLLM config. Cached.

    Call this before anything reads `r_cap()` / `p_cap()`. Pure host-side Python: it reads
    the config and does integer arithmetic, so it is safe to call from the MoE forward, but
    it should land on an EAGER call (it does -- the first MoE forward is the profiling run,
    long before capture).

    The caps derive from the config and nothing else -- there is no environment
    override. A wrong CAP faults as an illegal memory access rather than erroring
    gracefully, so the derived values are logged on every boot.
    """
    global _CAPS, _CAPS_TOPK
    if _CAPS is not None:
        if topk != _CAPS_TOPK:
            raise RuntimeError(
                f"ncompass: MoE top-k changed after the transport caps were fixed "
                f"({_CAPS_TOPK} -> {topk}). The caps size CUDA-graph-pinned buffers and "
                f"cannot be resized."
            )
        return _CAPS

    if _SNAPSHOT is None:
        raise RuntimeError(
            "ncompass: snapshot_config() has not run. The transport caps derive from VllmConfig, "
            "and get_current_vllm_config() is only valid inside a set_current_vllm_config() "
            "context -- i.e. during model construction, NOT during the forward pass. "
            "snapshot_config() must be called from the MoE method's __init__."
        )
    max_batch, tp, dp = _SNAPSHOT["max_batch"], _SNAPSHOT["tp"], _SNAPSHOT["dp"]

    p_derived = -(-max_batch // tp) * topk          # ceil(max_batch/tp) * topk
    r_derived = ep_world * p_derived

    # The launcher's independent formula, kept only as a cross-check on the derivation.
    r_legacy = dp * max_batch * topk

    from vllm.logger import init_logger

    init_logger(__name__).info(
        "ncompass caps: P_CAP=%d R_CAP=%d "
        "[max_batched=%d tp=%d dp=%d topk=%d ep_world=%d legacy_r=%d]",
        p_derived, r_derived,
        max_batch, tp, dp, topk, ep_world, r_legacy,
    )
    if r_legacy != r_derived:
        init_logger(__name__).warning(
            "ncompass caps: the launcher formula DP*max_batched*topk=%d differs from "
            "ep_world*P_CAP=%d; TP does not divide max_num_batched_tokens. Using the "
            "latter, which is what the slotted layout requires.",
            r_legacy, r_derived,
        )

    _CAPS, _CAPS_TOPK = (r_derived, p_derived), topk
    return _CAPS


def r_cap() -> int | None:
    return _CAPS[0] if _CAPS is not None else None


def p_cap() -> int | None:
    return _CAPS[1] if _CAPS is not None else None
