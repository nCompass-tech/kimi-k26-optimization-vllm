#!/usr/bin/env bash
# Compile the IBGDA device wrapper to a relocatable object, next to this script.
# Run at image build time: the object is nvlinked into every Triton kernel cubin at JIT
# time, so it must match the CUDA and NVSHMEM in the image that will run it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NVSHMEM_HOME="${NVSHMEM_HOME:-/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem}"
ARCH="${IBGDA_WRAPPER_ARCH:-sm_90a}"
nvcc -rdc=true -dc -Xcompiler -fPIC \
     "$HERE/nvshmem_wrapper.cu" -o "$HERE/nvshmem_wrapper.o" \
     -I"$NVSHMEM_HOME/include" -arch="$ARCH"
echo "built $HERE/nvshmem_wrapper.o (arch=$ARCH)"
nm "$HERE/nvshmem_wrapper.o" | grep -ciE 'ibgda|putmem' | xargs echo "ibgda/putmem symbols:"
