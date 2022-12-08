#include <SFML/Graphics/PrimitiveType.h>
#include <SFML/Graphics/RenderWindow.h>
#include <SFML/Graphics/Types.h>
#include <SFML/Graphics/VertexArray.h>
#include <SFML/Window/VideoMode.h>
#include <SFML/Window/Window.h>

#include <time.h>
#include <cufft.h>
#include <stdio.h>

#include "../utils/utils.cu"

#define GSIZE 1024
#define KSIZE 11 
#define RBYTES GSIZE*GSIZE*sizeof(cufftReal)

#define BOSCO_B1 34
#define BOSCO_B2 45
#define BOSCO_S1 34
#define BOSCO_S2 58


cufftReal* get_kernel_in_grid(int ksize, int gsize) {
    cufftReal* hgrid = (cufftReal*) calloc(sizeof(cufftReal), gsize*gsize);

    for(int i = -ksize/2; i < (ksize%2 ? ksize/2+1 : ksize/2); ++i)
        for(int j = -ksize/2; j < (ksize%2 ? ksize/2+1 : ksize/2); ++j)
            hgrid[(i>=0?i:gsize+i)*gsize+(j>=0?j:gsize+j)] = 1;

    return hgrid;
}

cufftReal* get_u10_grid(int gsize, int seed) {
    srand(seed);
    cufftReal* hgrid = (cufftReal*) calloc(sizeof(cufftReal), gsize*gsize);

    for (int i = 0; i < gsize*gsize; ++i)
        hgrid[i] = rand()%2;

    return hgrid;
}


void device_rfft2(cufftReal* in, cufftComplex* out, int size) {
    cufftHandle plan;
    cufftPlan2d(&plan, size, size, CUFFT_R2C);
    cufftExecR2C(plan, in, out);
    cufftDestroy(plan);
}

void device_irfft2(cufftComplex* in, cufftReal* out, int size) {
    cufftHandle plan;
    cufftPlan2d(&plan, size, size, CUFFT_C2R);
    cufftExecC2R(plan, in, out);
    cufftDestroy(plan);
}

__global__ void bosco_growth(cufftReal* grid, cufftReal* neigh, int b1, int b2, int s1, int s2, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int c = grid[i];
    int n = neigh[i];
    grid[i] = max(0,min(1, c + ((n >= b1) & (n <= b2)) - ((n < s1) | (n > s2))));
}

int main(int argc, char* argv[]) {

    int seed = time(0);
    printf("seed: %d;\n", seed);

    sfRenderWindow* window = sfRenderWindow_create((sfVideoMode){1000, 1000, 32}, "game of life", sfResize | sfClose, NULL);
    if (!window) return EXIT_FAILURE;
    
    // create vertex buffer
    sfVertexArray* vertex_array = sfVertexArray_create();
    sfVertexArray_setPrimitiveType(vertex_array, sfPoints);
    for (int i = 0; i < GSIZE*GSIZE; ++i) {
        sfVertex vertex;
        vertex.color = sfBlack;
        vertex.position = (sfVector2f){(float) i / GSIZE, (float) (i % GSIZE)};
        sfVertexArray_append(vertex_array, vertex);
    }

    cufftComplex* tmp = (cufftComplex*) malloc(sizeof(cufftComplex)*GSIZE*GSIZE);

    //initialize kernel
    cufftReal* kgrid = get_kernel_in_grid(KSIZE, GSIZE);
    cufftReal* ggrid = get_u10_grid(GSIZE, seed);
    
    cufftReal*    device_neigh     = NULL;
    cufftReal*    device_kgrid     = NULL;
    cufftReal*    device_ggrid     = NULL;
    cufftComplex* device_kgrid_fft = NULL;
    cufftComplex* device_ggrid_fft = NULL;

    cudaMalloc(&device_neigh     ,sizeof(cufftReal)    * GSIZE * GSIZE);
    cudaMalloc(&device_kgrid     ,sizeof(cufftReal)    * GSIZE * GSIZE);
    cudaMalloc(&device_ggrid     ,sizeof(cufftReal)    * GSIZE * GSIZE);
    cudaMalloc(&device_kgrid_fft ,sizeof(cufftComplex) * GSIZE * GSIZE);
    cudaMalloc(&device_ggrid_fft ,sizeof(cufftComplex) * GSIZE * GSIZE);

    cudaMemcpy(device_kgrid, kgrid, sizeof(cufftReal) * GSIZE * GSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_ggrid, ggrid, sizeof(cufftReal) * GSIZE * GSIZE, cudaMemcpyHostToDevice);

    device_rfft2(device_kgrid, device_kgrid_fft, GSIZE);
    
    float avg_clock = 0;
    for (int n = 0;; ++n) {
        clock_t start = clock();

        device_rfft2(device_ggrid, device_ggrid_fft, GSIZE);
        scaled_hadamart_product<<<dim3(GSIZE),dim3(GSIZE)>>>(device_ggrid_fft, device_kgrid_fft, GSIZE);
        device_irfft2(device_ggrid_fft, device_neigh, GSIZE); 
        bosco_growth<<<dim3(GSIZE),dim3(GSIZE)>>>(device_ggrid, device_neigh, BOSCO_B1, BOSCO_B2, BOSCO_S1, BOSCO_S2, GSIZE);

        for (int i = 0; i < GSIZE*GSIZE; ++i) {
            sfVertex* vertex = sfVertexArray_getVertex(vertex_array, i);
            vertex->color = round(ggrid[i]) ? sfWhite : sfBlack;
        }

        sfRenderWindow_drawVertexArray(window, vertex_array, NULL);
        sfRenderWindow_display(window);

        cudaMemcpy(ggrid, device_ggrid, sizeof(cufftReal)*GSIZE*GSIZE, cudaMemcpyDeviceToHost);

        // take time
        int msec = ((clock() - start) * 1000 / CLOCKS_PER_SEC)%1000;
        if(n > 100) avg_clock = (msec + (n-100) * avg_clock) / (n+1-100);
        printf("\rmsec: %d, avg:%f.", msec, avg_clock);
        fflush(stdout);

    }

    cudaFree(device_neigh);
    cudaFree(device_ggrid);
    cudaFree(device_kgrid);
    cudaFree(device_kgrid_fft);
    cudaFree(device_ggrid_fft);

    free(kgrid);
    free(ggrid);
    free(tmp);
    return 0;
}
