#include <stdio.h>
#include <cuda.h>

__global__ void bcast(int arg) 
{
    int laneId = threadIdx.x & 0x1f;
    int value;

    //if (laneId == 0)  value = arg;
    //if (laneId == 1)  value = arg+1;

    printf("Thread %d laneId %d.\n", threadIdx.x, laneId);
    printf("Thread %d value %d.\n", threadIdx.x, value);
    // Synchronize all threads in warp, and get "value" from lane 0
    //value = __shfl_sync(0xffffffff, value, 1);
    printf("Thread %d value %d arg %d.\n", threadIdx.x, value, arg);
    value = 1;
    if (value != arg)
        printf("Thread %d failed.\n", threadIdx.x);
}

int main07bcast()
{
    bcast<<< 1, 32 >>>(1234);
    cudaDeviceSynchronize();

    return 0;
}
