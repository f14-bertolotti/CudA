#include <SFML/Graphics/PrimitiveType.h>
#include <SFML/Graphics/RenderWindow.h>
#include <SFML/Graphics/Types.h>
#include <SFML/Graphics/VertexArray.h>
#include <SFML/Window/VideoMode.h>
#include <SFML/Window/Window.h>
#include <time.h>
#include <cstdlib>
#include <cufft.h>
#include <stdio.h>

#define SIZE 1024
#define SEED 1
#define CBYTES SIZE*SIZE*sizeof(cufftComplex)

cufftComplex* getGOLKernelFFT() {
    // define host game of life kernel
    cufftComplex* hgrid = (cufftComplex*) calloc(SIZE*SIZE, sizeof(cufftComplex));
    hgrid[1]              .x = 1.0;
    hgrid[SIZE-1]         .x = 1.0;
    hgrid[SIZE]           .x = 1.0;
    hgrid[SIZE+1]         .x = 1.0;
    hgrid[2*SIZE-1]       .x = 1.0;
    hgrid[SIZE*(SIZE-1)]  .x = 1.0;
    hgrid[SIZE*(SIZE-1)+1].x = 1.0;
    hgrid[SIZE*SIZE-1]    .x = 1.0;

    // copy host kernel to device
    cufftComplex* dgrid;
    cudaMalloc(&dgrid, CBYTES);
    cudaMemcpy(dgrid, hgrid, CBYTES, cudaMemcpyHostToDevice);

    // run fft on device
    cufftHandle planc2c;
    cufftPlan2d(&planc2c, SIZE, SIZE, CUFFT_C2C);
    cufftExecC2C(planc2c, dgrid, dgrid, CUFFT_FORWARD);

    // free plan and host resources
    cufftDestroy(planc2c);
    free(hgrid);

    return dgrid;
}

cufftComplex* getGOLGrid() {
    // get random host grid
    srand(SEED);
    cufftComplex* hgrid = (cufftComplex*) calloc(SIZE*SIZE, sizeof(cufftComplex));
    for (int i = 0; i < SIZE*SIZE; ++i) hgrid[i].x = rand() % 2;

    // copy host grid to device
    cufftComplex* dgrid;
    cudaMalloc(&dgrid, CBYTES);
    cudaMemcpy(dgrid, hgrid, CBYTES, cudaMemcpyHostToDevice);

    // free host resources
    free(hgrid);

    return dgrid;
 }

__global__ void emmul(cufftComplex* A, cufftComplex* B) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float x = (A[i].x*B[i].x - A[i].y*B[i].y);
    float y = (A[i].x*B[i].y + A[i].y*B[i].x);
    A[i].x = x/(SIZE*SIZE);
    A[i].y = y/(SIZE*SIZE);
}

__global__ void growth(cufftComplex* neighbours, cufftComplex* grid) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int n = round(neighbours[i].x);
    int c = round(grid[i].x);
    grid[i].x = max(0, min(1, c + 0 + (n == 3) - ((n < 2) || (n > 3))));
}


int main(int argc, char* argv[]) {

    // create window
    sfRenderWindow* window = sfRenderWindow_create((sfVideoMode){800, 600, 32}, "game of life", sfResize | sfClose, NULL);
    if (!window) return EXIT_FAILURE;
    
    // create vertex buffer
    sfVertexArray* vertex_array = sfVertexArray_create();
    sfVertexArray_setPrimitiveType(vertex_array, sfPoints);
    for (int i = 0; i < SIZE*SIZE; ++i) {
        sfVertex vertex;
        vertex.color = sfBlack;
        vertex.position = (sfVector2f){(float) i / SIZE, (float) (i % SIZE)};
        sfVertexArray_append(vertex_array, vertex);
    }

    // init game of life and kernel
    cufftComplex* hgrid  = (cufftComplex*) calloc(SIZE*SIZE, sizeof(cufftComplex));
    cufftComplex* kernel = getGOLKernelFFT();
    cufftComplex* grid   = getGOLGrid();
    cufftComplex* neigh  = getGOLGrid();

    cufftHandle plan;
    cufftPlan2d(&plan, SIZE, SIZE, CUFFT_C2C);

    // main loop
    float avg_clock = 0;
    for(int n = 0;; ++n) {
        // start timer
        clock_t start = clock();

        cufftExecC2C(plan, grid, neigh, CUFFT_FORWARD);
        emmul<<<dim3(SIZE*4), dim3(256)>>> (neigh, kernel);
        cufftExecC2C(plan, neigh, neigh, CUFFT_INVERSE);
        growth<<<dim3(SIZE*4), dim3(256)>>> (neigh, grid);

        for (int i = 0; i < SIZE*SIZE; ++i) {
            sfVertex* vertex = sfVertexArray_getVertex(vertex_array, i);
            vertex->color = round(hgrid[i].x) ? sfWhite : sfBlack;
        }

        sfRenderWindow_drawVertexArray(window, vertex_array, NULL);
        sfRenderWindow_display(window);

        cudaMemcpy(hgrid, grid, CBYTES, cudaMemcpyDeviceToHost);

        // take time
        int msec = ((clock() - start) * 1000 / CLOCKS_PER_SEC)%1000;
        if(n > 100) avg_clock = (msec + (n-100) * avg_clock) / (n+1-100);
        printf("\rmsec: %d, avg:%f.", msec, avg_clock);
        fflush(stdout);
    }


    cufftDestroy(plan);
    cudaFree(kernel);
    cudaFree(grid);
    cudaFree(neigh);
    free(hgrid);

    return 0;
}

