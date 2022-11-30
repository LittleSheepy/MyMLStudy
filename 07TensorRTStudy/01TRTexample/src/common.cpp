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
inline const char* severity_string(nvinfer1::ILogger::Severity t) {
    switch (t) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR:   return "error";
    case nvinfer1::ILogger::Severity::kWARNING: return "warning";
    case nvinfer1::ILogger::Severity::kINFO:    return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
    default: return "unknow";
    }
}
void TRTLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    if (severity <= Severity::kINFO) {
        // 打印带颜色的字符，格式如下：
        // printf("\033[47;33m打印的文本\033[0m");
        // 其中 \033[ 是起始标记
        //      47    是背景颜色
        //      ;     分隔符
        //      33    文字颜色
        //      m     开始标记结束
        //      \033[0m 是终止标记
        // 其中背景颜色或者文字颜色可不写
        // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
        if (severity == Severity::kWARNING) {
            printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
        }
        else if (severity <= Severity::kERROR) {
            printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
        }
        else {
            printf("%s: %s\n", severity_string(severity), msg);
        }
    }
}