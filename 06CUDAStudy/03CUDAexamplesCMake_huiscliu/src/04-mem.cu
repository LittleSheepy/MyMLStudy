#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>

typedef double FLOAT;
__global__ void sum(FLOAT* x)
{
    int tid = threadIdx.x;
    printf(">>>>>sumt id = %d\n", tid);
    x[tid] += 1;
}

int mem04()
{
    int N = 4;
    int nbytes = N * sizeof(FLOAT);

    FLOAT *dx = NULL, *hx = NULL;
    int i;

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);

    if (dx == NULL) {
        printf("couldn't allocate GPU memory\n");
        return -1;
    }

    /* alllocate CPU host mem: memory copy is faster than malloc */
    hx = (FLOAT*)malloc(nbytes);

    if (hx == NULL) {
        printf("couldn't allocate CPU memory\n");
        return -2;
    }

    /* init */
    printf(">>>>>hx original: \n");
    for (i = 0; i < N; i++) {
        hx[i] = i;
        printf("%g\n", hx[i]);
    }
    printf("<<<<<hx original: \n");

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

    /* call GPU */
    sum<<<1, N>>>(dx);

    /* let GPU finish */
    cudaDeviceSynchronize();

    /* copy data from GPU */
    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);


    printf("\nhx from GPU: \n");
    for (i = 0; i < N; i++) {
        printf("%g\n", hx[i]);
    }

    cudaFree(dx);
    free(hx);

    return 0;
}


int mem04_host()
{
    int N = 4;
    int nbytes = N * sizeof(FLOAT);

    FLOAT* dx = NULL, * hx = NULL;
    int i;

    /* allocate GPU mem */
    cudaMalloc((void**)&dx, nbytes);

    if (dx == NULL) {
        printf("couldn't allocate GPU memory\n");
        return -1;
    }

    /* alllocate CPU host mem: memory copy is faster than malloc */
    //hx = (FLOAT*)malloc(nbytes);
    cuMemAllocHost((void**)&hx, nbytes);

    if (hx == NULL) {
        printf("couldn't allocate CPU memory\n");
        return -2;
    }

    /* init */
    printf(">>>>>hx original: \n");
    for (i = 0; i < N; i++) {
        hx[i] = i;
        printf("%g\n", hx[i]);
    }
    printf("<<<<<hx original: \n");

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

    /* call GPU */
    sum << <1, N >> > (dx);

    /* let GPU finish */
    cudaDeviceSynchronize();

    /* copy data from GPU */
    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);

    /* �첽copy�Ļ� �͵ü�cudaDeviceSynchronize ͬ��*/
    //cudaMemcpyAsync(hx, dx, nbytes, cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();


    printf("\nhx from GPU: \n");
    for (i = 0; i < N; i++) {
        printf("%g\n", hx[i]);
    }

    cudaFree(dx);
    //free(hx);
    cudaFreeHost(hx);

    return 0;
}
