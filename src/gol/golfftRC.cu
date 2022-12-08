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
#define FBYTES SIZE*SIZE*sizeof(cufftReal)

cufftComplex* getGOLKernelFFT() {

    // define host game of life kernel
    cufftReal* hgrid = (cufftReal*) calloc(SIZE*SIZE, sizeof(cufftReal));
    hgrid[1]               = 1.0;
    hgrid[SIZE-1]          = 1.0;
    hgrid[SIZE]            = 1.0;
    hgrid[SIZE+1]          = 1.0;
    hgrid[2*SIZE-1]        = 1.0;
    hgrid[SIZE*(SIZE-1)]   = 1.0;
    hgrid[SIZE*(SIZE-1)+1] = 1.0;
    hgrid[SIZE*SIZE-1]     = 1.0;

    // copy host kernel to device
    cufftComplex* cgrid;
    cufftReal*    rgrid;
    cudaMalloc(&rgrid, FBYTES);
    cudaMemcpy(rgrid, hgrid, FBYTES, cudaMemcpyHostToDevice);
    cudaMalloc(&cgrid, CBYTES);
    cudaMemset(cgrid, 0, CBYTES);

    // run fft on device
    cufftHandle planR2C;
    cufftPlan2d(&planR2C, SIZE, SIZE, CUFFT_R2C);
    cufftExecR2C(planR2C, rgrid, cgrid);

    // free resources
    cufftDestroy(planR2C);
    cudaFree(rgrid);
    free(hgrid);

    return cgrid;
}

cufftReal* getGOLGrid() {
    // get random host grid
    srand(SEED);
    cufftReal* hgrid = (cufftReal*) calloc(SIZE*SIZE, sizeof(cufftReal));
    for (int i = 0; i < SIZE*SIZE; ++i) hgrid[i] = rand() % 2;

    // copy host grid to device
    cufftReal* dgrid;
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

__global__ void growth(cufftReal* neighbours, cufftReal* grid) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int n = round(neighbours[i]);
    int c = round(grid[i]);
    grid[i] = max(0, min(1, c + 0 + (n == 3) - ((n < 2) || (n > 3))));
}


int main(int argc, char* argv[]) {

    // create window
    sfRenderWindow* window = sfRenderWindow_create((sfVideoMode){1000, 1000, 32}, "game of life", sfResize | sfClose, NULL);
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
    cufftReal*    hgrid  = (cufftReal*)   calloc(SIZE*SIZE, sizeof(cufftReal));
    cufftComplex* cgrid  = (cufftComplex*)calloc(SIZE*SIZE, sizeof(cufftComplex));
    cufftComplex* kernel = getGOLKernelFFT();
    cufftReal*    grid   = getGOLGrid();
    
    cufftComplex* cneigh = NULL;
    cufftReal*    rneigh = NULL;
    cudaMalloc(&rneigh, sizeof(cufftReal   )*SIZE*SIZE);
    cudaMalloc(&cneigh, sizeof(cufftComplex)*SIZE*SIZE);
    cudaMemset(rneigh, 0, FBYTES);
    cudaMemset(cneigh, 0, CBYTES);

    cufftHandle planR2C;
    cufftHandle planC2R;
    cufftPlan2d(&planR2C, SIZE, SIZE, CUFFT_R2C);
    cufftPlan2d(&planC2R, SIZE, SIZE, CUFFT_C2R);

    // main loop
    float avg_clock = 0;
    for(int n = 0;; ++n) {
        // start timer
        clock_t start = clock();

        cufftExecR2C(planR2C, grid, cneigh);
        emmul<<<dim3(SIZE*8), dim3(128)>>> (cneigh, kernel);
        cufftExecC2R(planC2R, cneigh, rneigh);
        growth<<<dim3(SIZE*8), dim3(128)>>> (rneigh, grid);

        for (int i = 0; i < SIZE*SIZE; ++i) {
            sfVertex* vertex = sfVertexArray_getVertex(vertex_array, i);
            vertex->color = round(hgrid[i]) ? sfWhite : sfBlack;
        }

        sfRenderWindow_drawVertexArray(window, vertex_array, NULL);
        sfRenderWindow_display(window);

        cudaMemcpy(hgrid, grid, FBYTES, cudaMemcpyDeviceToHost);

        // take time
        int msec = ((clock() - start) * 1000 / CLOCKS_PER_SEC)%1000;
        if(n > 100) avg_clock = (msec + (n-100) * avg_clock) / (n+1-100);
        printf("\rmsec: %d, avg:%f.", msec, avg_clock);
        fflush(stdout);
    }


    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
    cudaFree(kernel);
    cudaFree(grid);
    cudaFree(rneigh);
    cudaFree(cneigh);
    free(hgrid);

    return 0;
}

