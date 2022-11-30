
// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>
// system include
#include <stdio.h>
int build_model();
int hello_inference();
int CNNUseAPI();

int main(){
    //build_model();
    //hello_inference();
    CNNUseAPI();
    return 0;
}