"""IBGDA-capable NVSHMEM device calls for Triton, WITHOUT torch's IBGDA-free libnvshmem_device.bc.

Root cause (confirmed): torch's `_nvshmem_triton` links the prebuilt `libnvshmem_device.bc`, which is
compiled with ZERO IBGDA device code (`nm .bc | grep ibgda` = 0; the `.a` has it). So its device puts
only drive the IBRC/proxy path -> under an IBGDA runtime the GPU-issued put faults. Triton-distributed's
fix (docs): don't IR-link the `.bc`; instead call thin `*_wrapper` device fns and `nvlink` the kernel
cubin against the REAL `libnvshmem_device.a` (+ a compiled wrapper obj), pulling in the IBGDA SASS.

This module replicates that mechanism on stock Triton 3.6.0 (has knobs.runtime.{add_stages_inspection_hook,
jit_post_compile_hook}) + reuses torch's nvshmem bootstrap (symm_mem) and torch's _nvshmemx_cumodule_init.

Usage:
    import ibgda_shmem as ib
    ib.install()                          # once, before compiling any wrapper kernel
    @triton.jit
    def k(...): ib.putmem_signal_block(dest, src, nbytes, sig, val, op, pe)   # NOT torch nvshmem.*
"""
import os
import re
import subprocess
import tempfile

import triton
import triton.language as tl
from triton.language import core as _core
from triton import knobs

_NV = os.environ.get("NVSHMEM_HOME",
                     "/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem")
_LIB = os.path.join(_NV, "lib")
_HERE = os.path.dirname(os.path.abspath(__file__))
_WRAPPER_O = os.environ.get("IBGDA_WRAPPER_O", os.path.join(_HERE, "nvshmem_wrapper.o"))
_NVLINK = os.environ.get("NVLINK", "/usr/local/cuda/bin/nvlink")


# ---------------------------------------------------------------------------
# Triton extern wrappers -> the *_wrapper device symbols (resolved by nvlink, NOT by a .bc).
# Same extern_elementwise pattern as torch's putmem_block wrapper; pointers cast to int64 by caller.
def _ext(symbol, arg_dtypes, ret="int32"):
    @_core.extern
    def _fn(*args, _semantic=None):
        return _core.extern_elementwise(
            "", "", list(args),
            {tuple(_core.dtype(d) for d in arg_dtypes): (symbol, _core.dtype(ret))},
            is_pure=False, _semantic=_semantic,
        )
    return _fn


_put_block_ext = _ext("nvshmemx_putmem_block_wrapper", ["int64", "int64", "int64", "int32"])
_put_nbi_block_ext = _ext("nvshmemx_putmem_nbi_block_wrapper", ["int64", "int64", "int64", "int32"])
_put_warp_ext = _ext("nvshmemx_putmem_warp_wrapper", ["int64", "int64", "int64", "int32"])
_putmem_ext = _ext("nvshmem_putmem_wrapper", ["int64", "int64", "int64", "int32"])
_put_signal_block_ext = _ext("nvshmemx_putmem_signal_block_wrapper",
                             ["int64", "int64", "int64", "int64", "int64", "int32", "int32"])
_fence_ext = _ext("nvshmem_fence_wrapper", [])
_fence_t0_ext = _ext("nvshmem_fence_t0_wrapper", [])
_quiet_ext = _ext("nvshmem_quiet_wrapper", [])
_signal_wait_ext = _ext("nvshmem_signal_wait_until_wrapper", ["int64", "int32", "int64"])
# CA-2: intra-node P2P pointer (int64 return). local_ptr(int64), my_pe(int32), dst_pe(int32) -> peer VA or 0.
_get_p2p_ptr_ext = _ext("nvshmemi_get_p2p_ptr_wrapper", ["int64", "int32", "int32"], ret="int64")


@triton.jit
def putmem_block(dest, src, nbytes, pe):
    _put_block_ext(dest.to(tl.int64), src.to(tl.int64), nbytes.to(tl.int64), pe)


@triton.jit
def putmem_nbi_block(dest, src, nbytes, pe):
    # NON-BLOCKING block put: posts the WQE and returns; completion is deferred to quiet().
    _put_nbi_block_ext(dest.to(tl.int64), src.to(tl.int64), nbytes.to(tl.int64), pe)


@triton.jit
def putmem_warp(dest, src, nbytes, pe):
    _put_warp_ext(dest.to(tl.int64), src.to(tl.int64), nbytes.to(tl.int64), pe)


@triton.jit
def putmem_signal_block(dest, src, nbytes, sig, signal, sig_op, pe):
    # pointers/nbytes -> int64; signal/sig_op/pe are plain ints (extern_elementwise coerces, like pe).
    _put_signal_block_ext(dest.to(tl.int64), src.to(tl.int64), nbytes.to(tl.int64),
                          sig.to(tl.int64), signal, sig_op, pe)


@triton.jit
def fence():
    _fence_ext()


@triton.jit
def fence_t0():
    # __syncthreads + single-thread nvshmem_fence + __syncthreads (see nvshmem_wrapper.cu).
    # Only valid at a BLOCK-UNIFORM call site.
    _fence_t0_ext()


@triton.jit
def quiet():
    _quiet_ext()


@triton.jit
def signal_wait_until(sig, cmp_op, cmp_val):
    _signal_wait_ext(sig.to(tl.int64), cmp_op, cmp_val.to(tl.int64))


@triton.jit
def get_p2p_ptr(local_ptr, my_pe, dst_pe):
    # CA-2: NVLink-mapped peer VA for local_ptr on dst_pe, or 0 if dst_pe is not P2P-reachable (use IBGDA).
    return _get_p2p_ptr_ext(local_ptr.to(tl.int64), my_pe, dst_pe)


# ---------------------------------------------------------------------------
# Compile-stage hook: ONLY for kernels whose PTX references a *_wrapper symbol -- compile PTX to a
# RELOCATABLE object (ptxas -c) and nvlink it against libnvshmem_device.a + the wrapper obj. Every
# OTHER kernel (Marlin, etc. in the serve) falls through to Triton's ORIGINAL cubin stage untouched.
def _nvlink_cubin(src, capability):
    from triton.backends.nvidia.compiler import sm_arch_from_capability, get_ptxas
    ptxas = get_ptxas(capability).path
    arch = sm_arch_from_capability(capability)
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".ptx") as fsrc:
        fsrc.write(src)
        fsrc.flush()
        fptx = fsrc.name
    fobj = fptx + ".o"
    fcomb = fobj + ".combined.cubin"
    try:
        r = subprocess.run([ptxas, "-c", "-lineinfo", "-v", f"--gpu-name={arch}", fptx, "-o", fobj],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ptxas -c failed:\n{r.stderr}")
        r = subprocess.run([_NVLINK, f"-arch={arch}", f"-L{_LIB}", "-lnvshmem_device",
                            fobj, _WRAPPER_O, "-o", fcomb], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvlink failed (arch={arch}):\n{r.stderr}")
        with open(fcomb, "rb") as f:
            return f.read()
    finally:
        for p in (fptx, fobj, fcomb):
            try:
                os.remove(p)
            except OSError:
                pass


def _stages_hook(backend, stages, options, language, capability):
    orig_cubin = stages["cubin"]

    def cubin(src, metadata):
        if re.search(r"nvshmem\w*wrapper\b", src):
            return _nvlink_cubin(src, capability)
        return orig_cubin(src, metadata)   # Triton's exact normal path for all non-wrapper kernels

    stages["cubin"] = cubin


def _post_compile_hook(*args, **kwargs):
    # After a wrapper kernel is compiled, register its CUmodule with NVSHMEM so device state
    # (incl. IBGDA QP state) is populated. Reuses torch's binding.
    if os.environ.get("IBGDA_SKIP_CUMODULE"):   # compile-only local validation (nvshmem not inited)
        return
    from torch._C._distributed_c10d import _nvshmemx_cumodule_init
    try:
        fn = kwargs["fn"].jit_function
        device = kwargs["compile"]["device"]
        kernel = fn.device_caches[device][0].get(kwargs["key"], None)
        if kernel is None:
            return
        if "nvshmem" in kernel.asm.get("ptx", ""):
            kernel._init_handles()
            _nvshmemx_cumodule_init(kernel.module)
    except Exception as e:  # noqa: BLE001
        import sys
        print(f"[ibgda_shmem] cumodule_init hook error: {type(e).__name__}: {e}", file=sys.stderr)


_installed = False


def install():
    global _installed
    if _installed:
        return
    assert os.path.exists(_WRAPPER_O), (
        f"ncompass: IBGDA wrapper object missing at {_WRAPPER_O} -- it is a BUILD OUTPUT, not a "
        f"tracked file. Compile it with ncompass_ibgda/build_wrapper.sh; both docker/Dockerfile.ncompass "
        f"and ncompass/docker/Dockerfile do this at image-build time. It must match the CUDA/NVSHMEM "
        f"of the image that runs it. (Single line on purpose: the multi-line version was truncated "
        f"by the worker log exactly when it mattered.)"
    )
    knobs.runtime.add_stages_inspection_hook = _stages_hook
    knobs.runtime.jit_post_compile_hook = _post_compile_hook
    _installed = True
