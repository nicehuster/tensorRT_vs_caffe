#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvCaffeParser.h"  
#include "common.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;
static Logger gLogger;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const std::vector<std::string> directories{ "data/samples/mnist/", "data/mnist/" };
std::string locateFile(const std::string& input)
{
    return locateFile(input, directories);
}

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& fileName, uint8_t buffer[INPUT_H*INPUT_W])
{
    readPGMFile(fileName, buffer, INPUT_H, INPUT_W);
}

void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile, directories).c_str(),
															  locateFile(modelFile, directories).c_str(),
															  *network,
															  DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv)
{
	
	// create a GIE model from the caffe model and serialize it to a stream
    IHostMemory *gieModelStream{nullptr};
   	caffeToGIEModel("mnist.prototxt", "mnist.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream);


	// parse the mean file
	ICaffeParser* parser = createCaffeParser();
	IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mnist_mean.binaryproto", directories).c_str());
	parser->destroy();
	const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());

	// deserialize the engine 
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
    if (gieModelStream) gieModelStream->destroy();

	IExecutionContext *context = engine->createExecutionContext();


	const std::string image_folder=argv[1];

	for(int i=0;i<10;i++){
		const std::string image_name=image_folder+"/"+std::to_string(i)+".png";
		cv::Mat mat=cv::imread(image_name,0);
		if(!mat.data){std::cerr<<"image read failed: "<<image_name;exit(0);}

		cv::resize(mat,mat,cv::Size(INPUT_W,INPUT_H));
		mat.convertTo(mat,CV_32FC1);
		const float*p=(float*)mat.data;

		//subtract it from the image
		float data[INPUT_H*INPUT_W];
		for (int i = 0; i < INPUT_H*INPUT_W; i++)
			data[i] = p[i]-meanData[i];
		// run inference
		float prob[OUTPUT_SIZE];
		long t0=cv::getTickCount();
		doInference(*context, data, prob, 1);
		float val{0.0f};
    	int idx{0};
		for (unsigned int i = 0; i < 10; i++){
        val = std::max(val, prob[i]);
        if (val == prob[i]) idx = i;
    	}
    	std::cout<<image_name<<":,predicted value: "<<idx<<", probability: "<<val<<std::endl;
		long t1=cv::getTickCount();
		double secs=(t1-t0)/cv::getTickFrequency();
		std::cout<<"********tensorRT takes "<<secs*1000<<"ms  **********"<<std::endl;
	}

	meanBlob->destroy();
	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	return 0;
}
