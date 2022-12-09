
/* asum: sum of all entries of a vector.
 * This code only calculates one block to show the usage of shared memory and synchronization */

#include <stdio.h>
#include <cuda.h>

typedef double FLOAT;

/* sum all entries in x and asign to y */
__device__ void warpReduce(volatile FLOAT *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}


__global__ void VecSumKnl_v2(const FLOAT *x, FLOAT *y)
{
    __shared__ FLOAT sdata[256];
    int tid = threadIdx.x;

    /* load data to shared mem */
    sdata[tid] = x[tid];
    __syncthreads();

    /* reduction using shared mem */
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();

    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32) warpReduce(sdata, tid);
    __syncthreads();

    if (tid == 0) y[0] = sdata[0];
}

__global__ void VecSumKnl_v1(const FLOAT *x, FLOAT *y)
{
    __shared__ FLOAT sdata[256];
    int tid = threadIdx.x;
    printf(">>>VecSumKnl_v1 tid=%d\n", tid);

    /* load data to shared mem */
    sdata[tid] = x[tid];
    __syncthreads();

    /* reduction using shared mem */
    if (tid < 128) {
        printf(">>>VecSumKnl_v1 ===tid <128===tid=%d,sdata[tid]=%f,sdata[tid + 128]=%f,\n", tid, sdata[tid], sdata[tid + 128]);
        sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();

    if (tid < 64) {
        printf(">>>VecSumKnl_v1 ===tid <64===tid=%d,sdata[tid]=%f,sdata[tid + 64]=%f,\n", tid, sdata[tid], sdata[tid + 64]);
        sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();

    if (tid < 32) {
        printf(">>>VecSumKnl_v1 ===tid <32===tid=%d,sdata[tid]=%f,sdata[tid + 32]=%f,\n", tid, sdata[tid], sdata[tid + 32]);
        sdata[tid] += sdata[tid + 32];
    }
    __syncthreads();

    if (tid < 16) {
        printf(">>>VecSumKnl_v1 ===tid <16===tid=%d,sdata[tid]=%f,sdata[tid + 16]=%f,\n", tid, sdata[tid], sdata[tid + 16]);
        sdata[tid] += sdata[tid + 16];
    }
    __syncthreads();

    if (tid < 8) {
        printf(">>>VecSumKnl_v1 ===tid <8===tid=%d,sdata[tid]=%f,sdata[tid + 8]=%f,\n", tid, sdata[tid], sdata[tid + 8]);
        sdata[tid] += sdata[tid + 8];
    }
    __syncthreads();

    if (tid < 4) {
        printf(">>>VecSumKnl_v1 ===tid <4===tid=%d,sdata[tid]=%f,sdata[tid + 4]=%f,\n", tid, sdata[tid], sdata[tid + 4]);
        sdata[tid] += sdata[tid + 4];
    }
    __syncthreads();

    if (tid < 2) {
        printf(">>>VecSumKnl_v1 ===tid <2===tid=%d,sdata[tid]=%f,sdata[tid + 2]=%f,\n", tid, sdata[tid], sdata[tid + 2]);
        sdata[tid] += sdata[tid + 2];
    }
    __syncthreads();

    if (tid == 0) {
        printf(">>>VecSumKnl_v1 ===tid ==0===tid=%d\n", tid);
        *y = sdata[0] + sdata[1];
    }
}

int reduction_1B_v1_06()
{
    int N = 256;   /* must be 256 */
    int nbytes = N * sizeof(FLOAT); // 2048=256*8

    FLOAT *dx = NULL, *hx = NULL;
    FLOAT *dy = NULL;
    int i;
    FLOAT as = 0;

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, sizeof(FLOAT));

    if (dx == NULL || dy == NULL) {
        printf("couldn't allocate GPU memory\n");
        return -1;
    }

    printf("allocated %e MB on GPU\n", nbytes / (1024.f * 1024.f));

    /* alllocate CPU mem */
    hx = (FLOAT *) malloc(nbytes);

    if (hx == NULL) {
        printf("couldn't allocate CPU memory\n");
        return -2;
    }
    printf("allocated %e MB on CPU\n", nbytes / (1024.f * 1024.f));

    /* init */
    for (i = 0; i < N; i++) {
        hx[i] = 1;
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

    /* call GPU */
    VecSumKnl_v1 <<<1, N>>>(dx, dy);

    /* let GPU finish */
    cudaDeviceSynchronize();

    /* copy data from GPU */
    cudaMemcpy(&as, dy, sizeof(FLOAT), cudaMemcpyDeviceToHost);

    printf("VecSumKnl, answer: 256, calculated by GPU:%g\n", as);

    cudaFree(dx);
    cudaFree(dy);
    free(hx);

    return 0;
}
