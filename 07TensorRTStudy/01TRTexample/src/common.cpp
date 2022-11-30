#include "common.h"
#include <stdio.h>
nvinfer1::Weights make_weights(float* ptr, int n) {
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}

vector<unsigned char> load_file(const string& file) {
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}
void TRTLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    if (severity <= Severity::kVERBOSE) {
        printf("=%d: %s\n", severity, msg);
    }
}