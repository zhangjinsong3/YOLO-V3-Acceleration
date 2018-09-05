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

#ifndef TENSORRT_CLASSIFIER_H
#define TENSORRT_CLASSIFIER_H

#include <algorithm>
#include <iomanip>
#include <thread>
#include <iterator>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include "classifier.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

static const int MAX_BUFFERS_ = 10;


// Logger for GIE info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
};

class Int8Calibrator : public IInt8EntropyCalibrator
{
public:
	Int8Calibrator(std::string calibrationTableFile)
	: calibrationTableFile_(calibrationTableFile) {}

	~Int8Calibrator() {}

	int getBatchSize() const override {
		return 0;
	}

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
		return false;
	}

	const void* readCalibrationCache(size_t& length) override
	{
		vCalibrationCache_.clear();
		std::ifstream input(calibrationTableFile_.c_str(), std::ios::binary);
		input >> std::noskipws;
		if (input.good())
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(vCalibrationCache_));

		length = vCalibrationCache_.size();
		return length ? &vCalibrationCache_[0] : nullptr;
	}

	void writeCalibrationCache(const void* cache, size_t length) override
	{
		std::cout << "writeCalibrationCache is called!" << std::endl;
	}

private:
	std::string calibrationTableFile_;
	std::vector<char> vCalibrationCache_;
};

class TensorRTClassifier  : public IClassifier {
public:
	TensorRTClassifier(const char *deployFile, // caffe prototxt file
						const char *modelFile, // trained caffe model
						const char *meanFile,  // mean file
						const std::string& inputs,
						const std::vector<std::string >& outputs,
						const int maxBatchSize,
						const int devID,
	                    nvcaffeparser1::IPluginFactory* pPluginFactory = nullptr,
                        std::string table = std::string());
	
	~TensorRTClassifier();

	void caffeToTensorRTModel(const char *deployFile,
								const char *modelFile,
								ICaffeParser *parser);
	void initInfer();
	
	// override
	void setInputData(float *pBGR,
						const int nWidth,
						const int nHeight,
						const int nBatchSize) override;
	
	void forward(INFER_OUTPUT_PARAMS *) override;
	
	int getInferWidth() const override;
	
	int getInferHeight() const override;
	
	std::vector<float > getMeanValues() const override;

private:
	int devID_;
	int maxBatchSize_;

	// tensorRT params
	ICudaEngine 		*pEngine_ 		= nullptr;
	ICaffeParser 		*pCaffeParser_ 	= nullptr;
	IBinaryProtoBlob 	*pMeanBlob_ 	= nullptr;
	IExecutionContext 	*pContext_ 		= nullptr;

	std::string inputBlobName_;
	std::vector<std::string > vOutputBlobNames_;
	nvcaffeparser1::IPluginFactory* pPluginFactory_{ nullptr };	// factory for plugin layers
	Int8Calibrator *pCalibrator_{ nullptr };
	std::string calibrationTable_;
	
	int 	nInputs_;
	int 	inputIndex_;
	DimsCHW	inputDim_;
	size_t 	inputSize_;
	
	int 	nOutputs_;
	std::vector<int > vOutputIndexs_;
	std::vector<DimsCHW > vOutputDims_; 
	std::vector<size_t > vOutputSizes_; 
	
	void	*apBuffers_[MAX_BUFFERS_]; // input and output buffer
    std::vector<float> vMeanValues_{0.f, 0.f, 0.f};
	
    // tensorRT logger
	Logger 	logger_; 
};


#endif // TENSORRT_CLASSIFIER_H
