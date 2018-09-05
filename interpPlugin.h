#ifndef INTERP_PLUGIN_H
#define INTERP_PLUGIN_H

#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include <stdio.h>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

#define BLOCK 512
#define ZOOM 2 // upsample *2

void interp_gpu(const float *x, int w, int h, int c, int batch, int zoomFactor, float *out, cudaStream_t stream);

template<int zoomFactor>
class Interp : public IPlugin
{
public:
	Interp() {}
	Interp(const void* buffer, size_t size)
	{
		// assert(size == sizeof(mInputSize));
		// mInputSize = *reinterpret_cast<const size_t*>(buffer);
		assert(size == sizeof(mInputDims));
		mInputDims = *reinterpret_cast<const Dims*>(buffer);
	}
	~Interp() {}

	// @ when creating the network
	int getNbOutputs() const override
	{
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		assert(nbInputDims == 1);
		assert(index == 0);
		assert(inputs[index].nbDims == 3);
        
		mOutputDims = DimsCHW(inputs[index].d[0], inputs[index].d[1] * zoomFactor, inputs[index].d[2] * zoomFactor);
		if (0) {
			std::cout << "IPlugin input dim = [" << inputs[index].d[0] << ", " << inputs[index].d[1]
				<< ", " << inputs[index].d[2] << "]" << std::endl;
			std::cout << "IPlugin output dim = [" << mOutputDims.d[0] << ", " << mOutputDims.d[1]
				<< ", " << mOutputDims.d[2] << "]" << std::endl;
		}
		return mOutputDims;
	}

	// @ when building the engine
	void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int maxBatchSize)	override
	{
		assert(1 == nbInputs && 1 == nbOutputs);
		mInputDims = inputs[0];
		mInputSize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
		// mOutputSize = outputs[0].d[0] * outputs[0].d[1] * outputs[0].d[2] * sizeof(float);
	}
	size_t getWorkspaceSize(int) const override
	{
		return 0;
	}

	// @ when serializing the engine
	size_t getSerializationSize() override
	{
		return sizeof(mInputDims);
	}
	void serialize(void* buffer) override
	{
		// *reinterpret_cast<size_t*>(buffer) = mInputSize;
		*reinterpret_cast<Dims*>(buffer) = mInputDims;
	}

	// @ when deserializing && executing the engine(at runtime)
	int initialize() override
	{
		return 0;
	}
	void terminate() override
	{
	}
	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
	{
		// TODO: why inputs idx 0?
		interp_gpu((const float*)inputs[0], mInputDims.d[2], mInputDims.d[1], mInputDims.d[0], batchSize, zoomFactor, (float *)outputs[0], stream); // TODO: didnt serialize mInputDims, can we use it? in that case, i serialized mInputDims, instead of mInputSize.
		return 0;
	}
	
protected:
	Dims mInputDims; //CHW
	Dims mOutputDims;
	size_t mInputSize;
	// size_t mOutputSize;
};


class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
	// @ when building the engine
	// caffe parser plugin implementation
	bool isPlugin(const char* layerName) override
	{
		return !(strcmp(layerName, "Interp85") && strcmp(layerName, "Interp97"));
	}
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
		assert(isPlugin(layerName));
		if (!strcmp(layerName, "Interp85"))
		{
			assert(layerName != "Interp85"); // debug_
			assert(mPluginInterp85.get() == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginInterp85 = std::unique_ptr<Interp<ZOOM>>(new Interp<ZOOM>());
			return mPluginInterp85.get();
		}
		else if (!strcmp(layerName, "Interp97"))
		{
			assert(layerName != "Interp97"); // debug_
			assert(mPluginInterp97.get() == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginInterp97 = std::unique_ptr<Interp<ZOOM>>(new Interp<ZOOM>());
			return mPluginInterp97.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	// @ at runtime
	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
		assert(isPlugin(layerName));
		if (!strcmp(layerName, "Interp85"))
		{
			assert(mPluginInterp85.get() == nullptr);
			mPluginInterp85 = std::unique_ptr<Interp<ZOOM>>(new Interp<ZOOM>(serialData, serialLength));
			return mPluginInterp85.get();
		}
		else if (!strcmp(layerName, "Interp97"))
		{
			assert(mPluginInterp97.get() == nullptr);
			mPluginInterp97 = std::unique_ptr<Interp<ZOOM>>(new Interp<ZOOM>(serialData, serialLength));
			return mPluginInterp97.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	void destroyPlugin()
	{
		//mPluginInterp97.release();		mPluginInterp97 = nullptr;
		//mPluginInterp85.release();		mPluginInterp85 = nullptr;
	}

	std::unique_ptr<Interp<ZOOM>> mPluginInterp85{ nullptr };
  std::unique_ptr<Interp<ZOOM>> mPluginInterp97{ nullptr };
};

#endif
