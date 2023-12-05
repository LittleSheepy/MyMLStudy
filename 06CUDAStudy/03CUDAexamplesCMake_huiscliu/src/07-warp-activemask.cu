#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <string>

// 获取激活状态的kernel的mask
__global__ void kernel()
{
    int oe = threadIdx.x & 0x1;  /* odd or even */
    unsigned int mask;

    /* sepatrate a warp to 2 sub-groups and generate masks */
    if (oe) {
        mask = __activemask();
    }
    else {
        mask = __activemask();
    }


    printf("Thread %d final mask = %ud \n", threadIdx.x,mask);
}

int main07activemask()
{
    kernel<<< 1, 8 >>>();
    cudaDeviceSynchronize();

    return 0;
}
/*
二进制         十进制
0000 0001         1
0000 0010         2
0000 0101         5
0000 1010         10
0001 0101         21
0010 1010         42
0101 0101         85
1010 1010         170

*/

// 线程数据广播
__global__ void bcast(int arg)
{
    int laneId = threadIdx.x & 0x1f;
    int value;

    if (laneId == 0)  value = arg;

    printf("Thread %d laneId %d.\n", threadIdx.x, laneId);
    printf("Thread %d value %d.\n", threadIdx.x, value);
    // Synchronize all threads in warp, and get "value" from lane 0
    value = __shfl_sync(0xffffffff, value, 0);
    printf("Thread %d value %d arg %d.\n", threadIdx.x, value, arg);
    if (value != arg)
        printf("Thread %d failed.\n", threadIdx.x);
}

int main07bcast()
{
    bcast << < 1, 32 >> > (1234);
    cudaDeviceSynchronize();

    return 0;
}


__global__ void scan4()
{
    int laneId = threadIdx.x & 0x1f;
    // Seed sample starting value (inverse of lane ID)
    int value = 31 - laneId;

    // Loop to accumulate scan within my partition.
    // Scan requires log2(n) == 3 steps for 8 threads
    // It works by an accumulated sum up the warp
    // by 1, 2, 4, 8 etc. steps.
    for (int i = 1; i <= 4; i *= 2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        int n = __shfl_up_sync(0xffffffff, value, i, 8);

        if ((laneId & 7) >= i) value += n;
    }

    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main07scan4()
{
    scan4 << < 1, 32 >> > ();
    cudaDeviceSynchronize();

    return 0;
}

