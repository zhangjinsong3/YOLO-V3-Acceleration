#include "cub/cub.cuh"
#include "common.h"

#include "bboxParser.h"

// nms involved two sorting, sorting for all bboxes (per Image) and sorting for all bboxes per class

/**
 *  This sort using cub, which divided into two step:
 *  1. determine temp_storage_bytes
 *  2. call the DeviceSegmentedRadixSort to sort.
 *
 *  The workspace includes two parts:
 *  a. offset array (nSegments + 1)
 *  b. temp storage for sorting
 *
 *  TODO:
 *  1. Since the array is same length, consider move the temp storage size outside for reuse.
 *  2. Add return value
 */
void sortScoresPerImage_gpu(
        const int   nBatch,
        const int   nItemsPerImage,
        void *      unsorted_scores,
        void *      unsorted_bbox_indices,
        void *      sorted_scores,
        void *      sorted_bbox_indices,
        void *      workspace,
        const size_t    maxSizeofWorkspaceInByte,
        cudaStream_t stream)
{
    // revese for the offset array
    void * d_offsets = workspace;
    void * cubWorkspace = nextWorkspacePtr((int8_t *)d_offsets, (nBatch + 1) * sizeof(int));

    // generated uniformed offsets (same stride)
    setUniformOffsets(nBatch, nItemsPerImage, (int *) d_offsets, stream);

    const int arrayLen = nBatch * nItemsPerImage;
    size_t temp_storage_bytes = cubSortPairsWorkspaceSize<float, int>(arrayLen, nBatch);

    // enough workspzce
    assert(temp_storage_bytes <= (maxSizeofWorkspaceInByte - (nBatch+1)*sizeof(int)));

    // sort with cub
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
            cubWorkspace,
            temp_storage_bytes,
            (const float *) (unsorted_scores),
            (float *) (sorted_scores),
            (const int *) (unsorted_bbox_indices),
            (int *) (sorted_bbox_indices),
            arrayLen,
            nBatch,
            (const int *) d_offsets,
            (const int *) d_offsets + 1,
            0,
            sizeof(float) * 8,
            stream);
}


/**
 * get bboxes data from prediction and archor (biases)
 */
__device__ box get_region_box(float x_in, float y_in,
                              float w_in, float h_in,
                              int i_in, int j_in,
                              float w, float h,
                              float biases_w, float biases_h)
{
    box b;
    b.x = (i_in + x_in) / w;
    b.y = (j_in + y_in) / h;
    b.w = exp(w_in) * biases_w / w;
    b.h = exp(h_in) * biases_h / h;
    return b;
}



/**
 * Prepare Data: split the output of region layer into two array:
 * 1. bbox: [batch][coords(4, no mask)][h][w]
 * 2. pred: [batch][classes][h][w]
 * in which, the pred is each to conf[box]*prob[class]
 *
 * Kernel launch size: n * h * w * nbbox
 */
#define nOutputLayer 3
template <unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
__global__ void splitOutputData_kernel(
        const int       nBatch,                    // batch
        const int       nClasses,
        const int       nBboxesPerLoc,    // #box
        const int       coords,                 // x,y,w,h
				const int				l0_w,
				const int				l0_h,
        const int       nCells,
        const bool      background,             // use background conf or not
        const bool      only_objectness,        // no class conf
        const float     thres,
        const float*    predictions,
        const float*    biases,
        float*          probes,
        box*            bboxes)
{
    size_t cur_idx  = blockIdx.x * nthds_per_cta + threadIdx.x;

    // for intput
    int bboxMemLen  = (nClasses + coords + 1) * nCells;
    int batchMemLen = nBboxesPerLoc           * bboxMemLen;

    int nBboxesPerImage = nBboxesPerLoc * nCells;
    int totalBboxes = nBatch * nBboxesPerImage;
    if (cur_idx < totalBboxes) {
        int batchIdx    = cur_idx / nBboxesPerImage;
        int bboxIdx     = (cur_idx % nBboxesPerImage) / (nCells);
        int locIdx      = (cur_idx % nBboxesPerImage) % (nCells);
				int locLayer, _cnt_offset = 1+2*2+4*4;
				for(int i = nOutputLayer-1; i >= 0; --i){
				  _cnt_offset -= (1<<i)*(1<<i); // zoomFactor = 2
				  if(locIdx >= _cnt_offset*l0_w*l0_h){
				    locLayer = i;
				    break;
				  }
				}

        // scale is the conf of bbox
        int scaleIdx    = batchIdx * batchMemLen
                          + bboxIdx * bboxMemLen
                          + coords * (nCells)           // 5th channel
                          + locIdx;
        float bboxConf  = predictions[scaleIdx];

        // output for probes
        int outProbBatchMemLen  = nBboxesPerLoc * nClasses * (nCells);

        // output of prob
        for (int i=0; i<nClasses; ++i){
            int classIdx = scaleIdx + (i+1) * nCells;
            float cur_prob = 0;
            // bbox_conf * class_prob
            if (!background){
                cur_prob = bboxConf * predictions[classIdx];
            }
            else{
                cur_prob = predictions[classIdx];
            }
            // bbox_conf only
            if (only_objectness){
                cur_prob = bboxConf;
            }
            // !!Note: the output prob order is: n -> classes -> box -> loc
            int probIdx = batchIdx * outProbBatchMemLen      // batch
                          + i * (nBboxesPerLoc * nCells)     // class
                          + bboxIdx * nCells                 // bbox
                          + locIdx;                          // loc
            probes[probIdx] = (cur_prob > thres) ? cur_prob : 0.f;
        }

        // batch * 4 * nCells
        int outBboxIdx  = batchIdx * nBboxesPerImage    // batch
                          + bboxIdx * nCells            // bbox
                          + locIdx;                     // loc
        int baseIdx = scaleIdx - coords * nCells;
        int subLocIdx = locIdx - _cnt_offset*l0_w*l0_h;
        int col = subLocIdx % ((1<<locLayer)*l0_w);
        int row = subLocIdx / ((1<<locLayer)*l0_w);
        bboxes[outBboxIdx] = get_region_box(predictions[baseIdx],
                                            predictions[baseIdx + nCells],
                                            predictions[baseIdx + 2 * nCells],
                                            predictions[baseIdx + 3 * nCells],
                                            col,            // column (i), nchw
                                            row,            // row (j), nchw
                                            (1<<locLayer)*l0_w,
                                            (1<<locLayer)*l0_h,
                                            biases[6*locLayer + bboxIdx*2],
                                            biases[6*locLayer + bboxIdx*2 + 1]);
    }
}


/**
 */
void splitOutputData_gpu(
        const int       nBatch,                    // batch
        const int       nClasses,
        const int       nBboxesPerLoc,    // #box
        const int       coords,                 // x,y,w,h
        const int       l0_w,
        const int				l0_h,
        const int       nCells,
        const bool      background,             // use background conf or not
        const bool      only_objectness,        // no class conf
        const float     thres,
        const float*    predictions,
        const float*    biases,
        float*          probes,
        box*            bboxes,
        cudaStream_t    stream)
{
    const int blockSize = 512;
    const int gridSize = (nBatch * nBboxesPerLoc * nCells + blockSize - 1) / blockSize;
    splitOutputData_kernel<blockSize>
        <<<gridSize, blockSize, 0, stream>>>
        (nBatch,
         nClasses,
         nBboxesPerLoc,
         coords,
         l0_w,
         l0_h,
         nCells,
         background,
         only_objectness,
         thres,
         predictions,
         biases,
         probes,
         bboxes);
}


__device__ box correct_region_box(box b, int w, int h, int netw, int neth)
{
    box b_new;
    int new_w = 0;
    int new_h = 0;

    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }

    b_new.x = (b.x - (netw - new_w)/2.f/netw) / ((float)new_w/netw);
    b_new.y = (b.y - (neth - new_h)/2.f/neth) / ((float)new_h/neth);
    b_new.w = b.w * (float)netw/new_w;
    b_new.h = b.h * (float)neth/new_h;

    return b_new;
}

/**
 * Correct bboxes with acutal image size.
 * The resize operation in YOLOv2 are
 *  1. Resize with original ratio, and keep long side the same size with net_input (w or h).
 *  2. Add black space in the short side equally.
 *  e.g. origial image = (832, 320), network input size = (416, 416).
 *       First resize the image to (416, 160), then adding 416*128 black pixels above and below the resized image.
 */
template <unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
__global__ void correct_region_boxes_kernel(
        const int       nBatch,                    // batch
        const int       nClasses,
        const int       nBboxesPerLoc,              // #box
        const int       nCells,
        const int       image_w,
        const int       image_h,
        const int       net_input_w,
        const int       net_input_h,
        box*            bboxes)
{
    size_t i  = blockIdx.x * nthds_per_cta + threadIdx.x;

    if (i < nBatch * nBboxesPerLoc * nCells){
        bboxes[i] = correct_region_box(bboxes[i], image_w, image_h, net_input_w, net_input_h);
    }
}

void correct_region_boxes_gpu(
        const int       nBatch,                    // batch
        const int       nClasses,
        const int       nBboxesPerLoc,    // #box
        const int       nCells,
        const int       image_w,
        const int       image_h,
        const int       net_input_w,
        const int       net_input_h,
        box*            bboxes,
        cudaStream_t    stream)
{
    const int blockSize = 512;
    const int gridSize = (nBatch * nBboxesPerLoc * nCells + blockSize - 1) / blockSize;

    correct_region_boxes_kernel<blockSize>
        <<<gridSize, blockSize, 0, stream>>> (
                nBatch,
                nClasses,
                nBboxesPerLoc,
                nCells,
                image_w,
                image_h,
                net_input_w,
                net_input_h,
                bboxes);
}


/**
 * The classes in previous output is ordered.
 *
 * With split data before, the data is splited into boxes and probes, and probes can be sorted per class
 *
 * * ! prob is in order batch -> classes -> box -> loc, so the interveal is nCells*nBboxesPerLoc
 * * output: only sorted index is needed and tmp_probes is needed
 *
 * TODO:
 * a. how to solve batched issue?
 *   * solution 1 (easiest): create temp GPU memory outside, use memory copy to put all prob together (as yolov2 darknet code)
 *   * solution 2 (easiest): create temp GPU memory in function (yolov2 darknet code)
 *   * solution 2: call cub sort batchSize times
 *   * solution 3 (best): use a larger arrayLen, skip first 5 channels (x, y, w, h, conf) for cub sort.
 * b. add return value
 **/
void sortScoresPerClass_gpu(
        const int       nBatch,
        const int       nClasses,
        const int       nBboxesPerImage,
        const void *    probes,
        void *          sorted_boxIdx,
        void *          workspace,
        const size_t    maxSizeofWorkspaceInByte,
        cudaStream_t    stream)
{
    // using solution 1
    const int nSegments  = nBatch * nClasses;
    const int arrayLen      = nBatch * nClasses * nBboxesPerImage;
    void * sorted_probes     = workspace;

    // initiate boxIndex
    void * unsorted_boxIdx   = nextWorkspacePtr((int8_t *)sorted_probes, arrayLen * sizeof(float));
    setUniformOffsets(arrayLen-1, 1, (int *)unsorted_boxIdx, stream);

    // initiate offset
    void *  d_offsets        = nextWorkspacePtr((int8_t *)unsorted_boxIdx, arrayLen * sizeof(int));
    setUniformOffsets(arrayLen, nBboxesPerImage, (int *)d_offsets, stream);

    // workspace
    size_t  cubOffsetSize   = (nSegments + 1) * sizeof(int);
    void * cubWorkspace     = nextWorkspacePtr((int8_t *)d_offsets, cubOffsetSize);

    size_t temp_storage_bytes =
        cubSortPairsWorkspaceSize
        <float, int>
        (arrayLen, nSegments);

    // enough temporary storage
    assert(   (arrayLen * sizeof(float)      // sorted_probes
                + arrayLen * sizeof(int)          // unsorted_boxIdx
                + cubOffsetSize                   // d_offsets
                + temp_storage_bytes )  <= maxSizeofWorkspaceInByte);

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
         cubWorkspace,
         temp_storage_bytes,
         (const float *) (probes),
         (float *) (sorted_probes),
         (const int *) (unsorted_boxIdx),
         (int *) (sorted_boxIdx),
         arrayLen,
         nSegments,
         (const int *)d_offsets,
         (const int *)d_offsets + 1,
         0,
         sizeof(float) * 8,
         stream);
}

__device__ float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1: l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

__device__ float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w<0 || h<0) return 0;
    float area = w*h;
    return area;
}

__device__ float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

__device__ float box_iou(box a, box b)
{
    return box_intersection(a, b) / box_union(a, b);
}


/**
 * nms for each classes
 * each block in charge of one class, in one batch
 *
 * 1. bboxes: n*nCells*nBboxesPerLoc
 * 2. probes, afterNMS_probes: n*nCells*nBboxesPerLoc * nClasses
 * 3. indexes, afterNMS_indexes: n*nCells*nBboxesPerLoc * nClasses
 *
 * One block handles one class.
 *
 * * "_indexes" or "_probes": the index of [0, n*nCells*nBboxesPerLoc*nClasses)
 * * "_bbox": the index of [0, n*nCells*nBboxesPerLoc)
 * * "_loc": the index of [0, nCells*nBboxesPerLoc)
 * * "_tSize": the index of [0, TSIZE)
 *
 */
template <int TSIZE>
__global__
void allClassNMS_kernel(
        const int       nBatch,        //batch
        const int       nClasses,
        const int       nBboxesPerLoc,
        const int       nCells,
        const float     nms_threshold,
        const box *     bboxes,
        const float *   probes,
        float *         afterNMS_probes,
        const int *     indexes,
        int *           afterNMS_indexes)
{
    // size: nCells*nBboxesPerLoc = (1+4+16)*13*13*3 = 10647 for YOLOv3
    extern __shared__ bool keptBboxInfoFlag_loc[];

    const int sizeOfClass = nCells * nBboxesPerLoc;
    const int classIdx = blockIdx.x % nClasses;
    const int batchIdx = blockIdx.x / nClasses;

    const int offset_indexes = batchIdx * nClasses * sizeOfClass + classIdx * sizeOfClass;
    const int maxIdx_indexes = offset_indexes + sizeOfClass;

    // the number of bbox is same for all classes
    int bboxIdx_tSize[TSIZE];
    box bbox_tSize[TSIZE];

    // initialize bbox, bboxInfo, kept_bboxinfo_flag
    #pragma unroll
    for (int t=0; t<TSIZE; ++t){
        const int curIdx_loc = threadIdx.x + t * blockDim.x;
        const int itemIdx_indexes = offset_indexes + curIdx_loc;

        if (itemIdx_indexes < maxIdx_indexes){
            // probes and indexes have same dimensions
            const int probIdx_probes = indexes[itemIdx_indexes];
            bboxIdx_tSize[t] = indexes[itemIdx_indexes];

            if (bboxIdx_tSize[t] != -1 && abs(probes[probIdx_probes])>1e-30){
                const int bboxIdx_bbox = batchIdx * sizeOfClass + bboxIdx_tSize[t] % sizeOfClass;

                bbox_tSize[t] = bboxes[bboxIdx_bbox];
                keptBboxInfoFlag_loc[curIdx_loc] = true;
            }
            else{
                keptBboxInfoFlag_loc[curIdx_loc] = false;
            }
        }
        else {
            keptBboxInfoFlag_loc[curIdx_loc] = false;
        }
    }
    __syncthreads();

    // filter out overlapped boxes with lower scores
    int refItemIdx_indexes = offset_indexes;    // first item
    int refBboxIdx_bbox = batchIdx * sizeOfClass + indexes[refItemIdx_indexes] % sizeOfClass;

    while (refItemIdx_indexes < maxIdx_indexes)
    {
        box refBbox = bboxes[refBboxIdx_bbox];

        for (int t=0; t<TSIZE; ++t){
            const int curIdx_loc = threadIdx.x + blockDim.x * t;
            const int itemIdx_indexes = offset_indexes + curIdx_loc;

            if ((keptBboxInfoFlag_loc[curIdx_loc]) && (itemIdx_indexes > refItemIdx_indexes)){
                if (box_iou(refBbox, bbox_tSize[t]) > nms_threshold){
                    keptBboxInfoFlag_loc[curIdx_loc] = false;
                }
            }
        }

        __syncthreads();

        do {
            refItemIdx_indexes ++;
        } while (!keptBboxInfoFlag_loc[refItemIdx_indexes - offset_indexes]
                 && refItemIdx_indexes < maxIdx_indexes);

        refBboxIdx_bbox = batchIdx * sizeOfClass + indexes[refItemIdx_indexes] % sizeOfClass;
    }

    // store data
    #pragma unroll
    for (int t=0; t<TSIZE; ++t){
        const int curIdx_loc = threadIdx.x + blockDim.x * t;
        const int readItemIdx_probes = indexes[offset_indexes + curIdx_loc];
        const int writeItemIdx_indexes = offset_indexes + curIdx_loc;

        if (readItemIdx_probes < maxIdx_indexes){
            afterNMS_probes[writeItemIdx_indexes]   = keptBboxInfoFlag_loc[curIdx_loc] ? probes[readItemIdx_probes] : 0.0f;
            afterNMS_indexes[writeItemIdx_indexes]  = keptBboxInfoFlag_loc[curIdx_loc] ? bboxIdx_tSize[t]: -1;
            /*afterNMS_probes[writeItemIdx_indexes]   = probes[readItemIdx_probes];*/
            /*afterNMS_indexes[writeItemIdx_indexes]  = bboxIdx_tSize[t];*/
            /*afterNMS_indexes[writeItemIdx_indexes]  = -1;*/
        }
    }
}


/**
 * TODO: add return type
 */
void allClassNMS_gpu(
        const int       nBatch,        //batch
        const int       nClasses,
        const int       nBboxesPerLoc,
        const int       nCells,
        const float     nms_threshold,
        void *          bboxes,
        void *          probes,
        void *          afterNMS_probes,
        void *          indexes,
        void *          afterNMS_indexes,
        cudaStream_t    stream)
{
#define P(tsize) allClassNMS_kernel<(tsize)>

    void (*kernel[32]) (const int,
                       const int,
                       const int,
                       const int,
                       const float,
                       const box   *,
                       const float  *,
                       float  *,
                       const int *,
                       int *)
    = {
        P(1), P(2), P(3), P(4), P(5), P(6), P(7), P(8), P(9), P(10), P(11), P(12), P(13), P(14), P(15), P(16), P(17), P(18), P(19), P(20), P(21), P(22), P(23), P(24), P(25), P(26), P(27), P(28), P(29), P(30), P(31), P(32)};

    const int blockSize = 512;
    const int gridSize = nClasses * nBatch;
    int t_size = (nCells * nBboxesPerLoc + blockSize - 1) / blockSize;

    kernel[t_size - 1]
        <<< gridSize, blockSize, blockSize * t_size * sizeof(bool), stream >>>
        ( nBatch,
          nClasses,
          nBboxesPerLoc,
          nCells,
          nms_threshold,
          (const box *)bboxes,
          (const float *)probes,
          (float *)afterNMS_probes,
          (const int *) indexes,
          (int *) afterNMS_indexes);
}



size_t getWorkspaceSizeInByte(
        const int       nBatch,
        const int       nClasses,
        const int       nBboxesPerLoc,
        const int       nCells)
{
    // 1. temporary storage of sortScoresPerClass_GPU
    size_t mem_for_sortScoresPerClass = 0;
    // 1.1 sorted probes
    mem_for_sortScoresPerClass = nBatch * nClasses * nBboxesPerLoc * nCells * sizeof(float);
    // 1.2 unsorted bbox
    mem_for_sortScoresPerClass += nBatch * nClasses * nBboxesPerLoc * nCells * sizeof(int);
    // 1.3 offset
    mem_for_sortScoresPerClass += (nBatch * nClasses + 1) * sizeof(int);
    // 1.4 cub workspace
    mem_for_sortScoresPerClass += cubSortPairsWorkspaceSize<float, int>(nBatch * nClasses * nBboxesPerLoc * nCells, nBatch * nClasses);

    // 2. temporary storage of sortScoresPerImage_gpu
    size_t mem_for_sortScoresPerImage = 0;
    // 2.1 offset
    mem_for_sortScoresPerImage = (nBatch + 1) * sizeof(int);
    // 2.2 cub workspace
    mem_for_sortScoresPerImage += cubSortPairsWorkspaceSize<float, int>(nBatch * nClasses * nBboxesPerLoc * nCells, nBatch);

    size_t maxWorkSpaceInByte = (mem_for_sortScoresPerClass > mem_for_sortScoresPerImage) ? mem_for_sortScoresPerClass : mem_for_sortScoresPerImage;

    return maxWorkSpaceInByte;
}
