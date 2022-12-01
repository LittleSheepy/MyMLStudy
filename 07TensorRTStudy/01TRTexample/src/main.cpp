
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
int yolox_detect();
int retinaface_detect();
int unet();
int chinese_classifer_bert();

int main(){
    // 更改控制台输出编码 —— 65001表示UTF-8编码格式
    SetConsoleOutputCP(65001);
    unet();
    return 0;
}