#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>

using namespace std;
bool exists(const string& path);
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);
nvinfer1::Weights make_weights(float* ptr, int n); 
vector<unsigned char> load_file(const string& file);
class TRTLogger : public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;
};
vector<string> load_labels(const char* file);




