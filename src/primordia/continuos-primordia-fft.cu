#include <SFML/Graphics.hpp>
#include <cstdint>
#include <cstdlib>
#include <cufft.h>

#include "../utils/utils.cu"

#define GSIZE 2048
#define STATES 13
#define TIME 10
#define SEED 1


cufftReal* get_random_grid(int states, int size, int seed) {
    srand(seed);
    cufftReal* grid = (cufftReal*) malloc(sizeof(cufftReal)*size*size);
    for (int i = 0; i < size * size; ++i) grid[i] = rand() % states; 
    return grid;
}

cufftReal* get_kernel(int states, int size) {
    cufftReal* kernel = (cufftReal*) calloc(size*size, sizeof(cufftReal));
    kernel[1]               = 1.0f/(8*(states-1));
    kernel[size-1]          = 1.0f/(8*(states-1));
    kernel[size]            = 1.0f/(8*(states-1));
    kernel[size+1]          = 1.0f/(8*(states-1));
    kernel[2*size-1]        = 1.0f/(8*(states-1));
    kernel[size*(size-1)]   = 1.0f/(8*(states-1));
    kernel[size*(size-1)+1] = 1.0f/(8*(states-1));
    kernel[size*size-1]     = 1.0f/(8*(states-1));
    return kernel;
}

__global__ void multistate_gol_growth(cufftReal* grid, cufftReal* neigh, int states, int time, int size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    cufftReal u = neigh[id];
    cufftReal a = grid[id];
    grid[id] = min((cufftReal) states-1,max(0.0f,a + (1.0f/time)*(((u>=0.20f)&(u<=0.25)) - ((u<=0.18)|(u>=0.33)))));
}

__global__ void colorize(uint8_t* color_field, cufftReal* grid, int states, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    uint8_t v = (uint8_t) round(grid[i]);
    color_field[4 * i + 0] = v * 255.0f/states;
	color_field[4 * i + 1] = v * 255.0f/states;
	color_field[4 * i + 2] = v * 255.0f/states;
	color_field[4 * i + 3] = 255;
}



int main (int argc, char *argv[]) { 
    sf::RenderWindow window(sf::VideoMode(GSIZE, GSIZE), "larger than life fft");
    sf::Texture texture;
	sf::Sprite sprite;
	std::vector<sf::Uint8> pixelBuffer(GSIZE * GSIZE * 4);
	texture.create(GSIZE, GSIZE);

    cufftHandle planR2C;
    cufftHandle planC2R;

    cufftPlan2d(&planR2C, GSIZE, GSIZE, CUFFT_R2C);
    cufftPlan2d(&planC2R, GSIZE, GSIZE, CUFFT_C2R);
    
    cufftReal*   grid    = get_random_grid(STATES, GSIZE, SEED);
    cufftReal* kernel    = get_kernel(STATES, GSIZE);

    uint8_t*      device_color_field    = NULL;
    cufftReal*    device_real_grid      = NULL;
    cufftReal*    device_real_neigh     = NULL;
    cufftReal*    device_real_kernel    = NULL;
    cufftComplex* device_complex_grid   = NULL;
    cufftComplex* device_complex_kernel = NULL;
    
    cudaMalloc(&device_real_grid     , sizeof(cufftReal)    * GSIZE * GSIZE);
    cudaMalloc(&device_real_neigh    , sizeof(cufftReal)    * GSIZE * GSIZE);
    cudaMalloc(&device_real_kernel   , sizeof(cufftReal)    * GSIZE * GSIZE);
    cudaMalloc(&device_complex_grid  , sizeof(cufftComplex) * GSIZE * GSIZE);
    cudaMalloc(&device_complex_kernel, sizeof(cufftComplex) * GSIZE * GSIZE);
    cudaMalloc(&device_color_field   , sizeof(uint8_t) * 4  * GSIZE * GSIZE);

    cudaMemcpy(device_real_grid  , grid  , sizeof(cufftReal) * GSIZE * GSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_real_kernel, kernel, sizeof(cufftReal) * GSIZE * GSIZE, cudaMemcpyHostToDevice);
    
    cufftExecR2C(planR2C, device_real_kernel, device_complex_kernel);

    float avg_clock = 0.0f;
    for (int n = 0;; ++n) {
        clock_t start = clock();


        cufftExecR2C(planR2C, device_real_grid, device_complex_grid);
        scaled_hadamart_product<<<dim3(GSIZE*GSIZE/256),dim3(256)>>>(device_complex_grid, device_complex_kernel, GSIZE);
        cufftExecC2R(planC2R, device_complex_grid, device_real_neigh);
        multistate_gol_growth<<<dim3(GSIZE*GSIZE/256), dim3(256)>>>(device_real_grid, device_real_neigh, STATES, TIME, GSIZE);
        colorize<<<dim3(GSIZE*GSIZE/256), dim3(256)>>>(device_color_field, device_real_grid, STATES, GSIZE);

        cudaMemcpy(pixelBuffer.data(), device_color_field, sizeof(uint8_t)*GSIZE*GSIZE*4, cudaMemcpyDeviceToHost);
        
        texture.update(pixelBuffer.data());
		sprite.setTexture(texture);
		sprite.setScale({2, 2});
		window.draw(sprite);
		window.display();

        int msec = ((clock() - start) * 1000 / CLOCKS_PER_SEC)%1000;
        avg_clock = (msec + (n) * avg_clock) / (n+1);
        printf("\rmsec: %03d, avg:%3.5f.", msec, avg_clock);
        fflush(stdout);
    }

    cufftDestroy(planR2C);
    cufftDestroy(planC2R);

    cudaFree(device_real_grid);
    cudaFree(device_real_neigh);
    cudaFree(device_real_kernel);
    cudaFree(device_complex_grid);
    cudaFree(device_complex_kernel);
    cudaFree(device_color_field);

    free(grid);
    free(kernel);
    return 0;
}
