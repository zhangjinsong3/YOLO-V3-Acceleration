#include <regionLayer.h>
#include <cfloat>

/** \brief kernel for softmax
 *  - n is the number of classes (included the background)
 *
 *  - The CPU implementation is
 *  for b in batch:
 *      for g in groups:
 *          softmax(input + b*batchOffset + g*groupOffset, n, temp, stride, output + b*batchOffset + g*groupOffset)
 *
 *  - The GPU implementation put the two for-loop into parallel.
 *
 *  - nthdsPerCTA: the max number of threads per block.
 *  - Each thread will in charge of one point softmax for all classes.
 *  - Total number of threads: batch * groups
 *
 *  - TODO: using warp shuffle instead of loop in one thread.
 */
template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
__global__ void softmaxKernel(const float * input,
            const int n,
            const int batch,
            const int batchOffset,
            const int groups,
            const int groupOffset,
            const int stride,
            const float temp,
            float * output)
{
    int id = blockIdx.x * nthdsPerCTA + threadIdx.x;

    // per batch, per group
    if (id < batch * groups)
    {
        int b = id / groups;
        int g = id % groups;
        float sum = 0.;
        float largest = -FLT_MAX;
        int offset = b*batchOffset + g*groupOffset;
        for (int i = 0; i < n; ++i)
        {
            float val = input[i*stride + offset];
            largest = (val > largest) ? val : largest;
        }
        for (int i = 0; i < n; ++i)
        {
            float e = exp(input[i*stride + offset]/temp - largest/temp); // bound score in (-inf,0], and denominator fractor in (0,1].
            sum += e;
            output[i*stride + offset] = e;
        }
        for (int i = 0; i < n; ++i)
          output[i*stride + offset] /= sum;
    }
}


/**
 * \brief Sigmoid function
 *
 * "__launch_bounds__" ensures the universality of kernel
 */
template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
__global__ void activateKernel(float * data,
            const int range)
{
    int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
    if (i < range)
      data[i] = 1. / (1. + exp(-data[i]));
}

/**
 * \brief region layer of YOLOv3
 * Includes activation and softmax.
 * - num: # bounding box per location
 *
 * If we integrated into tensorRT, we can use input and output are different memory.
 * If it is standalone GPU code (in main.cpp), we can use input and output the same buffer.
 *
 * Note: The elements in YOLOv3
 * * 4*nCells coords,
 * * nCells conf,
 * * classes*nCells classes
 *  e.g.
 *      * nCells for 0 class (background)
 *      * nCells for 1 class
 *      * ...
 */
void regionLayer_gpu(
        const int batch,
        const int C,
        const int nCells,
        const int num,
        const int coords,
        const int classes,
        const float * input,
        float * output,
        cudaStream_t stream)
{
    const int blockSize = 256;
    const int gridSize_Act1 = (2*nCells + blockSize - 1) / blockSize;  // x, y
    const int gridSize_Act2 = (nCells + blockSize - 1) / blockSize;    // conf
    const int gridSize_Softmax = (nCells + blockSize - 1) / blockSize;   // classes
    // for YOLOv3, the output of final layer is C*nCells, in which, C includes all the conf, coord, and claesses.

#ifdef REGION_IN_TRT
    // TRT, input and output are diff buffer
    ck(cudaMemcpy((void*)output, (void*)input, batch*C*nCells*sizeof(float), cudaMemcpyDeviceToDevice));
#endif
    // else input and output can be same buffer

    for (int b = 0; b < batch; ++b) {
        for (int n = 0; n < num; ++n) {
            // activate on (x,y)
            int index = b*C*nCells   // per batch
                        + n*nCells*(coords+classes+1);  // coords, classes and confidence
            activateKernel<blockSize>
                <<<gridSize_Act1, blockSize, 0, stream>>>
                (output + index, 2*nCells);

            // activate on probes on conf
            index = b*C*nCells
                    + n*nCells*(coords+classes+1)
                    + 4*nCells;                        // skip coords
            activateKernel<blockSize>
                <<<gridSize_Act2, blockSize, 0, stream>>>
                (output + index, nCells);

						// softmax for all classes
            index = b*C*nCells
                    + n*nCells*(coords+classes+1)
                    + 5*nCells;                        // skip conf
						softmaxKernel<blockSize>
						    <<<gridSize_Softmax, blockSize, 0, stream>>>
                (input + index,     // input: skip loc, conf
                 classes,           // n: #classes
                 batch*num,         // batch: batch * #bound_box
                 (C*nCells/num),    // batchOffset: number of bounding_box in total
                 nCells,            // groups
                 1,                 // groupOffset
                 nCells,            // stride
                 1.f,               // temp
                 output + index);   // output
        }
    }
}

#define nOutputLayer 3
template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
__global__ void reorgOutputKernel(
        const int       nBatch,
        const int       nClasses,
        const int       nBboxesPerLoc,
        const int       coords,
				const int				l0_w,
				const int				l0_h,
        const int       nCells,
        float*          dpData_unordered[],
        float*          dpData)
{
    long i = blockIdx.x * nthdsPerCTA + threadIdx.x;
    const int bboxMemLen  = (nClasses + coords + 1) * nCells;
    const int batchMemLen = nBboxesPerLoc * bboxMemLen;
    const long range = nBatch * batchMemLen;
    if (i < range) // voc<266175 coco<904995 wrt. 416*416 input
    {
        int b = i / batchMemLen;
        int bboxIdx = (i % batchMemLen) / bboxMemLen;
        int channelIdx = ((i % batchMemLen) % bboxMemLen) / nCells;
        int locIdx = (i % batchMemLen) % nCells;
        int locLayer, cnt_offset = 1+2*2+4*4;
				for(int j = nOutputLayer-1; j >= 0; --j){
				    cnt_offset -= (1<<j)*(1<<j); // zoomFactor = 2
				    if(locIdx >= cnt_offset*l0_w*l0_h){
				        locLayer = j;
				        break;
				    }
				}
				dpData[i] = dpData_unordered[locLayer]\
				                [b*nBboxesPerLoc*(nClasses+coords+1)*(1<<locLayer)*(1<<locLayer)*l0_w*l0_h +\
				                bboxIdx*(nClasses+coords+1)*(1<<locLayer)*(1<<locLayer)*l0_w*l0_h +\
				                channelIdx*(1<<locLayer)*(1<<locLayer)*l0_w*l0_h +\
				                locIdx - cnt_offset*l0_w*l0_h];
    }
}

void reorgOutput_gpu(
        const int       nBatch,
        const int       nClasses,
        const int       nBboxesPerLoc,
        const int       coords,
				const int				l0_w,
				const int				l0_h,
        const int       nCells,
        float*          dpData_unordered[],
        float*          dpData,
        const long      nData,
        cudaStream_t    stream)
{
    const int blockSize = 512;
    const int gridSize = (nData + blockSize - 1) / blockSize;
    reorgOutputKernel<blockSize>
        <<<gridSize, blockSize, 0, stream>>>
        (nBatch, nClasses, nBboxesPerLoc, coords, l0_w, l0_h, nCells, dpData_unordered, dpData);
    
}
