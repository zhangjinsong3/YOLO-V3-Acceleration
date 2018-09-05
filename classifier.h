/*
* Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

#ifndef CLASSIFIER_H
#define CLASSIFIER_H 

#include <vector>
#include "NvInfer.h"
#include <cuda_runtime.h>

using namespace nvinfer1;

typedef struct INFER_OUTPUT_PARAMS_ {
	int nBatchSize_;
	std::vector<float *> vpInferResults_;
	std::vector<int    > vnLens_;
    std::vector<DimsCHW > vOutputDims_;
} INFER_OUTPUT_PARAMS;

class IClassifier {
public:
	virtual void setInputData(float *pBGR,
								const int nWidth,
								const int nHeight,
								const int nBatchSize) = 0;
	
	virtual void forward(INFER_OUTPUT_PARAMS *) = 0;
	
	virtual int getInferWidth() const = 0;
	
	virtual int getInferHeight() const = 0;
	
	virtual std::vector<float > getMeanValues() const = 0;

protected:
	virtual ~IClassifier() {}
};

#endif


