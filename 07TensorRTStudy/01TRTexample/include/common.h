#pragma once
#include <NvInfer.h>
nvinfer1::Weights make_weights(float* ptr, int n); 
class TRTLogger : public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;
};





