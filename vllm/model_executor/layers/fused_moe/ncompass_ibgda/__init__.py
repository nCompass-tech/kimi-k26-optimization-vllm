# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IBGDA-capable NVSHMEM device calls for Triton.

Works around a packaging defect upstream of vLLM: torch's Triton/NVSHMEM integration
IR-links a prebuilt `libnvshmem_device.bc` that is compiled with **zero** IBGDA device code
(`nm` on the bitcode finds none; the static archive `libnvshmem_device.a` has it). Kernels
built that way only ever drive the IBRC/CPU-proxy path, and fault under an IBGDA runtime.

`ibgda_shmem` skips the bitcode entirely: `nvshmem_wrapper.cu` provides thin `extern "C"
__device__` shims, and a Triton hook nvlinks the kernel cubin against the real
`libnvshmem_device.a` plus that wrapper object, pulling the genuine IBGDA SASS into the
kernel.

`nvshmem_wrapper.o` is a BUILD OUTPUT, not a source file -- it is compiled from the `.cu`
next to it (see `build_wrapper.sh`) and is deliberately not tracked in git: it is
architecture-specific (`-arch=sm_90a`) and must match the CUDA/NVSHMEM in the image that
runs it.

The durable fix belongs upstream -- either torch ships a bitcode with IBGDA compiled in, or
vLLM performs the nvlink step in its own Triton integration. This module is a stopgap.
"""
