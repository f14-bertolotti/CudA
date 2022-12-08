#include <stdio.h>

__global__ void swap_top_bottom(int size, int* grid) {
    // It swaps top real row with bottom ghost row, and
    // it swaps top ghost rows with bottom real row.
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (0 < i && i < size-1) {
        grid[size * (size-1) + i] = grid[size + i]; // top <- bottom
        grid[i] = grid[size * (size-2) + i]; // bottom <- top
    }
}

__global__ void swap_left_right(int size, int* grid) {
    // It swaps left real row with right ghost row, and
    // it swaps left ghost rows with right real row.
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (0 <= i && i < size) {
        grid[i*size] = grid[i*size + size - 2]; // left <- right
        grid[i*size + size - 1] = grid[i * size + 1]; // right <- left
    }
}

__global__ void game_of_life(int size, int* grid, int* new_grid) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int id = iy * size + ix;
    int numNeighbors;

    if (0 < ix && ix < size-1 && 0 < iy && iy < size-1) {
        numNeighbors = grid[(iy-1) * size + ix+1] + grid[(iy+0) * size + ix+1] + grid[(iy+1) * size + ix+1] + 
                       grid[(iy-1) * size + ix+0] + 0                          + grid[(iy+1) * size + ix+0] +
                       grid[(iy-1) * size + ix-1] + grid[(iy+0) * size + ix-1] + grid[(iy+1) * size + ix-1]; 
        int cell = grid[iy * size + ix];
        new_grid[id] = max(0, min(1, cell + 0 + (numNeighbors==3) - ((numNeighbors < 2) || (numNeighbors > 3))));
    }

}


