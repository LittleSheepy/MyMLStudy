#ifndef REDUCTION_AUX
#define REDUCTION_AUX

#include <stdio.h>
#include <cuda.h>

/* get thread id: 1D block and 2D grid */
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)

/* get block id: 2D grid */
#define get_bid() (blockIdx.x + blockIdx.y * gridDim.x)

/* get time stamp */
double get_time(void);


#endif