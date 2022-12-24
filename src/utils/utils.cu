#include <cufft.h>

__global__ void scaled_hadamart_product(cufftComplex* A, cufftComplex* B, int scale, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        float x = (A[i].x*B[i].x - A[i].y*B[i].y);
        float y = (A[i].x*B[i].y + A[i].y*B[i].x);
        A[i].x = x/scale;
        A[i].y = y/scale;
    }
}

__global__ void game_of_life_growth(cufftReal* grid, cufftReal* neigh, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        int n = round(neigh[i]);
        int c = round(grid[i]);
        grid[i] = max(0, min(1, c + (n == 3) - ((n < 2) || (n > 3))));
    }
}

__global__ void larger_than_life_growth(cufftReal* grid, cufftReal* neigh, int b1, int b2, int s1, int s2, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        int c = grid[i];
        int n = neigh[i];
        grid[i] = max(0,min(1, c + ((n >= b1) & (n <= b2)) - ((n < s1) | (n > s2))));
    }
}

__global__ void primordia_growth(cufftReal* grid, cufftReal* neigh, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        float c = grid[i];
        float n = neigh[i];
        grid[i] = max(0.0f,min(1.0f, c + (1/10.0f) * (((n >= 0.20)&(n <= 0.25)) - ((n <= 0.19)|(n >= 0.33)))));
    }
}
