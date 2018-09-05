#include "common.h"

// alignptr
int8_t * alignPtr(int8_t * ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t)ptr;
    if (addr % to) {
        addr += to - addr % to;
    }
    return (int8_t *)addr;
}

// calc next ptr (consider alignment)
int8_t * nextWorkspacePtr(int8_t * ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t *)addr, CUDA_MEM_ALIGN);
}


template <unsigned nthds_per_cta>
__launch_bounds__ (nthds_per_cta)
__global__ void setUniformOffsets_kernel(
        const int   num_segments,
        const int   offset,
        int *       d_offsets)
{
    const int idx = blockIdx.x * nthds_per_cta + threadIdx.x;
    if (idx <= num_segments){
        d_offsets[idx] = idx * offset;
    }
}

void setUniformOffsets(
        const int       num_segments,
        const int       offset,
        int *           d_offsets,
        cudaStream_t    stream)
{
    const int blockSize = 32;
    const int gridSize = (num_segments + 1 + blockSize - 1) / blockSize;
    setUniformOffsets_kernel<blockSize>
        <<<gridSize, blockSize, 0, stream>>>
        (num_segments, offset, d_offsets);
}
