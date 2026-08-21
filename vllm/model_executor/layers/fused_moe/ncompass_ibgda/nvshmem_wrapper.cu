// Minimal IBGDA-capable nvshmem device wrapper for the deva2a transport.
// Compiled to a relocatable device object and nvlink'd (against libnvshmem_device.a) into the
// Triton kernel cubin -- this pulls in the REAL IBGDA device code (the .bc torch links has none).
// ALL wrappers return int (return 0) to match Triton extern_elementwise's int32 return type
// (nvlink checks prototypes exactly; a void return -> "prototype doesn't match").
#include <nvshmem.h>
#include <nvshmemx.h>
extern "C" {
__device__ int nvshmemx_putmem_block_wrapper(void* d, const void* s, size_t n, int pe){ nvshmemx_putmem_block(d,s,n,pe); return 0; }
__device__ int nvshmemx_putmem_nbi_block_wrapper(void* d, const void* s, size_t n, int pe){ nvshmemx_putmem_nbi_block(d,s,n,pe); return 0; }
__device__ int nvshmemx_putmem_warp_wrapper(void* d, const void* s, size_t n, int pe){ nvshmemx_putmem_warp(d,s,n,pe); return 0; }
__device__ int nvshmem_putmem_wrapper(void* d, const void* s, size_t n, int pe){ nvshmem_putmem(d,s,n,pe); return 0; }
__device__ int nvshmemx_putmem_signal_block_wrapper(void* d, const void* s, size_t n, uint64_t* sig, uint64_t val, int op, int pe){ nvshmemx_putmem_signal_block(d,s,n,sig,val,op,pe); return 0; }
__device__ int nvshmem_fence_wrapper(){ nvshmem_fence(); return 0; }
// SINGLE-THREAD fence, per the NVSHMEM spec's documented block-scoped put -> fence -> signal pattern:
// "after a block-scoped Put returns and all threads in the CTA synchronize with __syncthreads, a single
// thread in the CTA may call nvshmem_fence before issuing a signal... Not every issuing thread needs to
// call nvshmem_fence in this pattern." Triton's extern_elementwise makes EVERY thread execute the plain
// wrapper above (256 threads at num_warps=8), so this variant tests how much of the measured fence cost
// is redundant per-thread work. Both barriers are required and guard opposite sides of the fence:
//   leading  __syncthreads: makes "all threads have ISSUED the data put" visible to thread 0, since
//                           nvshmem_fence orders only the CALLING thread's operations.
//   trailing __syncthreads: stops threads 1..N-1 racing into the (block-scoped) flag put before
//                           thread 0's fence has executed.
// Call site must be block-uniform (it is: the dest predicate is scalar/uniform per program).
__device__ int nvshmem_fence_t0_wrapper(){
    __syncthreads();
    if (threadIdx.x == 0) nvshmem_fence();
    __syncthreads();
    return 0;
}
__device__ int nvshmem_quiet_wrapper(){ nvshmem_quiet(); return 0; }
__device__ int nvshmem_signal_wait_until_wrapper(uint64_t* sig, int cmp, uint64_t val){ nvshmem_signal_wait_until(sig,cmp,val); return 0; }
__device__ int nvshmemx_signal_op_wrapper(uint64_t* sig, uint64_t val, int op, int pe){ nvshmemx_signal_op(sig,val,op,pe); return 0; }
// CA-2: intra-node P2P pointer (DeepEP nvshmemi_get_p2p_ptr mechanism). Returns the NVLink-mapped
// peer VA for `local_ptr` on `dst_pe`, or 0 if dst_pe is NOT P2P-reachable (-> caller uses IBGDA).
__device__ uint64_t nvshmemi_get_p2p_ptr_wrapper(uint64_t local_ptr, int my_pe, int dst_pe){
    if (my_pe == dst_pe) return local_ptr;
    unsigned long long peer_base =
        __ldg((const unsigned long long*)nvshmemi_device_state_d.peer_heap_base_p2p + dst_pe);
    if (peer_base == 0ULL) return 0ULL;          // RDMA-only peer -> not P2P
    return (uint64_t)peer_base + (local_ptr - (uint64_t)nvshmemi_device_state_d.heap_base);
}
}
