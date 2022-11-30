#include "common.h"
#include <stdio.h>
nvinfer1::Weights make_weights(float* ptr, int n) {
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}

void TRTLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    if (severity <= Severity::kVERBOSE) {
        printf("=%d: %s\n", severity, msg);
    }
}