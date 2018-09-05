#ifndef COMMON_H_
#define COMMON_H_

#include <cub/cub.cuh>

#define CUDA_MEM_ALIGN 256

// alignptr
int8_t * alignPtr(int8_t * ptr, uintptr_t to);

int8_t * nextWorkspacePtr(int8_t * ptr, uintptr_t previousWorkspaceSize);

void setUniformOffsets(const int num_segments, const int offset, int * d_offsets, cudaStream_t stream);

/**
 * Determine the usage of temporary memory for cub sort
 * The cub::DeviceSegmentedRadixSort can be used for batched (segmented) sort.
 */
template <typename KeyT, typename ValueT>
size_t cubSortPairsWorkspaceSize(int num_items, int num_segments)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
	(void *)NULL, temp_storage_bytes,
	(const KeyT *)NULL, (KeyT *)NULL,
	(const ValueT *)NULL, (ValueT *)NULL,
	num_items,     // # items
	num_segments,  // # segments
	(const int *)NULL, (const int *)NULL);
    return temp_storage_bytes;
}

#endif
