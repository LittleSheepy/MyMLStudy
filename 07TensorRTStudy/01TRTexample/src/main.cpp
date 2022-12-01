
// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>
// system include
#include <stdio.h>
#include "Windows.h"
int build_model();
int hello_inference();
int CNNUseAPI();
int onnx_parser();
int hello_plugin();
int integrate_easyplugin();
int int8();
int plugin_3rd();
int full_cnn_classifier();
int yolov5_detect();

int main(){
    // 更改控制台输出编码 —— 65001表示UTF-8编码格式
    SetConsoleOutputCP(65001);
    //build_model();
    //hello_inference();
    //CNNUseAPI();
    //onnx_parser();
    //hello_plugin();
    //integrate_easyplugin();
    //int8();
    //plugin_3rd();
    //full_cnn_classifier();
    yolov5_detect();
    return 0;
}