#ifndef BBOX_PARSER_H
#define BBOX_PARSER_H

#include <vector>
// cub for sort
#include "regionLayer.h"

void sortScoresPerImage_gpu(
        const int   nBatch,
        const int   nItemsPerImage,
        void *      unsorted_scores,
        void *      unsorted_bbox_indices,
        void *      sorted_scores,
        void *      sorted_bbox_indices,
        void *      workspace,
        const size_t    maxSizeofWorkspaceInByte,
        cudaStream_t stream);

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
        cudaStream_t    stream);


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
        cudaStream_t    stream);


void sortScoresPerClass_gpu(
        const int       nBatch,
        const int       nClasses,
        const int       nBboxesPerLoc,
        const void *    probes,
        void *          sorted_boxIdx,
        void *          workspace,
        const size_t    maxSizeofWorkspaceInByte,
        cudaStream_t    stream);


void allClassNMS_gpu(
        const int       nBatch,                    //batch
        const int       nClasses,
        const int       nBboxesPerLoc,
        const int       nCells,
        const float     nms_threshold,
        void *          bboxes,
        void *          probes,
        void *          afterNMS_probes,
        void *          indexes,
        void *          afterNMS_indexes,
        cudaStream_t    stream);


size_t getWorkspaceSizeInByte(
        const int       nBatch,
        const int       nClasses,
        const int       nBboxesPerLoc,
        const int       nCells);

#endif
