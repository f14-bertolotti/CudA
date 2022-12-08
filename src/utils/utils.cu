#include <cufft.h>

__global__ void roll_left(cufftReal* grid, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int i  = id / size;
    int j  = id % size;
    int tmp = grid[id];
    __syncthreads();
    grid[i*size+(j==0?size-1:j-1)] = tmp;
}

__global__ void roll_up(cufftReal* grid, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int i  = id / size;
    int j  = id % size;
    int tmp = grid[id];
    __syncthreads();
    grid[(i==0?size-1:i-1)*size+j] = tmp;
}

__global__ void scaled_hadamart_product(cufftComplex* A, cufftComplex* B, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float x = (A[i].x*B[i].x - A[i].y*B[i].y);
    float y = (A[i].x*B[i].y + A[i].y*B[i].x);
    A[i].x = x/(size*size);
    A[i].y = y/(size*size);
}


