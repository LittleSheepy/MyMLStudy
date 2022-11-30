
// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream> 
#include <fstream> // 后面要用到ios这个库
#include <vector>
#include "common.h"
using namespace std;

bool build_model03(){
    TRTLogger logger;

    // ----------------------------- 1. 定义 builder, config 和network -----------------------------
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    // 构建一个模型
    /*
        Network definition:

        image
          |
        conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
          |
        relu
          |
        prob
    */


    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    const int num_input = 1;
    const int num_output = 1;
    float layer1_weight_values[] = {
        1.0, 2.0, 3.1, 
        0.1, 0.1, 0.1, 
        0.2, 0.2, 0.2
    }; // 行优先
    float layer1_bias_values[]   = {0.0};

    // 如果要使用动态shape，必须让NetworkDefinition的维度定义为-1，in_channel是固定的
    nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, num_input, -1, -1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 9);
    nvinfer1::Weights layer1_bias   = make_weights(layer1_bias_values, 1);
    auto layer1 = network->addConvolution(*input, num_output, nvinfer1::DimsHW(3, 3), layer1_weight, layer1_bias);
    layer1->setPadding(nvinfer1::DimsHW(1, 1));

    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kRELU); // *(layer1->getOutput(0))
     
    // 将我们需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    // 配置暂存存储器，用于layer实现的临时存储，也用于保存中间激活值
    config->setMaxWorkspaceSize(1 << 28);

    // --------------------------------- 2.1 关于profile ----------------------------------
    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();

    // 配置最小允许1 x 1 x 3 x 3
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, num_input, 3, 3));
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, num_input, 3, 3));

    // 配置最大允许10 x 1 x 5 x 5
    // if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, num_input, 5, 5));
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // -------------------------- 3. 序列化 ----------------------------------
    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("03engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;
}

void inference03(){
    // ------------------------------- 1. 加载model并反序列化 -------------------------------
    TRTLogger logger;
    auto engine_data = load_file("03engine.trtmodel");
    nvinfer1::IRuntime* runtime   = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    /*
        Network definition:

        image
          |
        conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
          |
        relu
          |
        prob
    */

    // ------------------------------- 2. 输入与输出 -------------------------------
    float input_data_host[] = {
        // batch 0
        1,   1,   1,
        1,   1,   1,
        1,   1,   1,

        // batch 1
        -1,   1,   1,
        1,   0,   1,
        1,   1,   -1
    };
    float* input_data_device = nullptr;

    // 3x3输入，对应3x3输出
    int ib = 2;
    int iw = 3;
    int ih = 3;
    //float output_data_host[ib * iw * ih];
    float output_data_host[18];
    float* output_data_device = nullptr;
    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);


    // ------------------------------- 3. 推理 -------------------------------
    // 明确当前推理时，使用的数据输入大小
    execution_context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, ih, iw));
    float* bindings[] = {input_data_device, output_data_device};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);


    // ------------------------------- 4. 输出结果 -------------------------------
    for(int b = 0; b < ib; ++b){
        printf("batch %d. output_data_host = \n", b);
        for(int i = 0; i < iw * ih; ++i){
            printf("%f, ", output_data_host[b * iw * ih + i]);
            if((i + 1) % iw == 0)
                printf("\n");
        }
    }

    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();
}

int CNNUseAPI(){

    if(!build_model03()){
        return -1;
    }
    inference03();
    return 0;
}