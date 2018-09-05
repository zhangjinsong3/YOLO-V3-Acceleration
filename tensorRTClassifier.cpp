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
#include "tensorRTClassifier.h"
#include "nvUtils.h"
#include <nvToolsExt.h>

TensorRTClassifier::TensorRTClassifier(const char *deployFile, // caffe prototxt file
										const char *modelFile, // trained caffe model
										const char *meanFile,  // mean file
										const std::string& inputs,
										const std::vector<std::string >& outputs,
										const int maxBatchSize,
										const int devID,
	                                    nvcaffeparser1::IPluginFactory* pPluginFactory,
                                        std::string table)
	: maxBatchSize_(maxBatchSize), devID_(devID)
{
	// set input and outputs of the neural network
	inputBlobName_ = inputs;
	vOutputBlobNames_ = outputs;
    pPluginFactory_ = pPluginFactory;
    calibrationTable_ = table;
	
    LOG_DEBUG(logger, "TensorRTClassifier: parse caffe model...");
    if(!calibrationTable_.empty()) {
		LOG_ERROR(logger, "Use INT8 calibration table");
	    pCalibrator_ = new Int8Calibrator(calibrationTable_);
	    assert(nullptr != pCalibrator_);
	}

    cudaSetDevice(devID_);
	pCaffeParser_ = createCaffeParser();
	caffeToTensorRTModel(deployFile, modelFile, pCaffeParser_);
	if (meanFile) {
		pMeanBlob_ = pCaffeParser_->parseBinaryProto(meanFile);
	}
	pContext_ = pEngine_->createExecutionContext();
	initInfer();
}

TensorRTClassifier::~TensorRTClassifier() {
	cudaSetDevice(devID_);
	
	pContext_->destroy();
	pEngine_->destroy();
	pCaffeParser_->destroy();
	
    if (nullptr != pCalibrator_) {
		delete pCalibrator_;
	}
	
	if (pMeanBlob_) {
		pMeanBlob_->destroy();
	}

	ck(cudaFree(apBuffers_[inputIndex_]));
	for (int i = 0; i < nOutputs_; ++i) {
		ck(cudaFree(apBuffers_[vOutputIndexs_[i]]));
	}
}

void TensorRTClassifier::setInputData(float *pBGR,
										const int nWidth,
										const int nHeight,
										const int nBatchSize) {
	assert(inputDim_.w()	== nWidth);
	assert(inputDim_.h() 	== nHeight);
	assert(maxBatchSize_ 	== nBatchSize);
	assert(inputSize_		== 3 * nWidth * nHeight * nBatchSize * sizeof(float));
	
	ck(cudaMemcpy(apBuffers_[inputIndex_], pBGR, inputSize_, cudaMemcpyHostToDevice));
}

void TensorRTClassifier::forward(INFER_OUTPUT_PARAMS *pInferOutputParams) {
	cudaSetDevice(devID_);
	pContext_->execute(maxBatchSize_, apBuffers_);
	
	pInferOutputParams->vpInferResults_.resize(nOutputs_, nullptr);	
	pInferOutputParams->vnLens_.resize(nOutputs_, 0);	
	pInferOutputParams->vOutputDims_.resize(nOutputs_);
	pInferOutputParams->nBatchSize_ = maxBatchSize_;
	//pInferOutputParams->nInferLen_ = outputDims_.c;
	//pInferOutputParams->dpInferResults_ = reinterpret_cast<float* >(apBuffers_[outputIndex_]);
	for (int j = 0; j < nOutputs_; ++j) {
		pInferOutputParams->vpInferResults_[j] = reinterpret_cast<float* >(apBuffers_[vOutputIndexs_[j]]);
		pInferOutputParams->vnLens_[j] 	= vOutputDims_[j].c() * vOutputDims_[j].h() * vOutputDims_[j].w();
	    pInferOutputParams->vOutputDims_[j] = vOutputDims_[j];
    }
}

int TensorRTClassifier::getInferWidth() const {
	return inputDim_.w();
}

int TensorRTClassifier::getInferHeight() const {
	return inputDim_.h();
}

std::vector<float > TensorRTClassifier::getMeanValues() const {
	return vMeanValues_;
}


void TensorRTClassifier::initInfer() {
	//LOG_DEBUG(logger, "TensorRTClassifier: init inference...");
	ck(cudaSetDevice(devID_));
	
	// input tensor param
	nInputs_	 = 1;
	inputIndex_  = pEngine_->getBindingIndex(inputBlobName_.c_str());
	inputDim_   = static_cast<DimsCHW&&>(pEngine_->getBindingDimensions(inputIndex_));
	inputSize_   = maxBatchSize_ * inputDim_.c() * inputDim_.h() *
					inputDim_.w() * sizeof(float);
	
	// output tensor param
	nOutputs_ = vOutputBlobNames_.size();
	vOutputIndexs_.resize(nOutputs_, -1);
	vOutputDims_.resize(nOutputs_, {0,0,0});
	vOutputSizes_.resize(nOutputs_, -1);

	for (int i = 0; i < nOutputs_; ++i) {
		vOutputIndexs_[i] = pEngine_->getBindingIndex(vOutputBlobNames_[i].c_str());
		vOutputDims_[i]  = static_cast<DimsCHW&&>(pEngine_->getBindingDimensions(vOutputIndexs_[i]));
		vOutputSizes_[i]  = maxBatchSize_ * vOutputDims_[i].c() * vOutputDims_[i].h() *
							vOutputDims_[i].w() * sizeof(float);
	}

	// allocate GPU buffers
	ck(cudaMalloc(&apBuffers_[inputIndex_], inputSize_));
	ck(cudaMemset(apBuffers_[inputIndex_], 0, inputSize_));
	
	for (int i = 0; i < nOutputs_; ++i) {
		ck(cudaMalloc(&apBuffers_[vOutputIndexs_[i]], vOutputSizes_[i]));
		ck(cudaMemset(apBuffers_[vOutputIndexs_[i]], 0, vOutputSizes_[i]));
	}
	
	//Dims4 dim;
	DimsNCHW dim;
	if (pMeanBlob_) {
		const float *meanData = reinterpret_cast<const float*>(pMeanBlob_->getData());
		dim = pMeanBlob_->getDimensions();
		
		float avg[3] = {0.0, 0.0, 0.0};	
		for (int i = 0; i < dim.c(); ++i) {
			for (int j = 0; j < dim.h() * dim.w(); ++j) {
				avg[i] += meanData[i*dim.h()*dim.w()+j];
			}
			avg[i] /= dim.w()*dim.h();
		}
		
		vMeanValues_[0] = avg[0];
		vMeanValues_[1] = avg[1];
		vMeanValues_[2] = avg[2];
	}
	
    LOG_DEBUG(logger, " ");
    LOG_DEBUG(logger, "=========== Network Parameters Begin ===========");
	LOG_DEBUG(logger, "Network Input:");
	LOG_DEBUG(logger, "	>Channel :" << inputDim_.c());
	LOG_DEBUG(logger, "	>Height  :" << inputDim_.h());
	LOG_DEBUG(logger, "	>Width   :" << inputDim_.w());
	for (int i = 0; i < nOutputs_; ++i) {
		LOG_DEBUG(logger, "Network Output [" << i << "] :" << vOutputBlobNames_[i]);
		LOG_DEBUG(logger, "	>Channel :" << vOutputDims_[i].c());
		LOG_DEBUG(logger, "	>Height  :" << vOutputDims_[i].h());
		LOG_DEBUG(logger, "	>Width   :" << vOutputDims_[i].w());
	}
	LOG_DEBUG(logger, "=========== Network Parameters End   ===========");
	
}


void TensorRTClassifier::caffeToTensorRTModel(const char *deployFile, const char *modelFile, ICaffeParser *parser)
{
	IBuilder* builder = createInferBuilder(logger_);
	INetworkDefinition* network = builder->createNetwork();
	if (nullptr != pPluginFactory_) {
		parser->setPluginFactory(pPluginFactory_);
	}
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile,
															  modelFile,
															  *network,
															  DataType::kFLOAT);
	for (auto& s : vOutputBlobNames_)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize_);
	builder->setMaxWorkspaceSize(16 << 20);
    if (nullptr != pCalibrator_) {
		builder->setInt8Mode(pCalibrator_ != nullptr);
		builder->setInt8Calibrator(pCalibrator_);
    }
	pEngine_ = builder->buildCudaEngine(*network);
	assert(pEngine_);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	builder->destroy();
}
