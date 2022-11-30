#pragma once
#include <NvInfer.h>
#include <vector>
#include <fstream>
using namespace std;
nvinfer1::Weights make_weights(float* ptr, int n); 
vector<unsigned char> load_file(const string& file);
class TRTLogger : public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;
};





