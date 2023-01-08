#include <cufft.h>

#include "../src/utils/kernels.cpp"
#include "../src/utils/grids.cpp"
#include "../src/utils/cufft.cpp"
#include "../src/utils/utils.cu"
#include "liblenia.h"


FFT* fft;
int gsize, gsize1, bsize1, gsize2, bsize2;
DeviceMatrix<cufftComplex>* device_cplx_grid;
DeviceMatrix<cufftReal   >* device_real_neigh;
DeviceMatrix<cufftComplex>* device_cplx_kernel;

void initialize(int size) {
    // gpu block and grid set up
    gsize = size;
    cudaOccupancyMaxPotentialBlockSize(&gsize1, &bsize1, scaled_hadamart_product, 0, size * size);
    cudaOccupancyMaxPotentialBlockSize(&gsize2, &bsize2,     game_of_life_growth, 0, size * size);
    gsize1 = ((size*size) + bsize1 - 1) / bsize1; 
    gsize2 = ((size*size) + bsize2 - 1) / bsize2; 
    fft = new FFT(size);

    // grid util matrix
    device_cplx_grid = new DeviceMatrix<cufftComplex>(size);

    // kernel util matrix
    Kernel kernel(bell, 13, 0.5, 0.15, size);
    DeviceMatrix<cufftReal> device_real_kernel(&kernel);
    device_cplx_kernel = new DeviceMatrix<cufftComplex>(size);
    fft->rfft2(&device_real_kernel, device_cplx_kernel);
    
    // neigh util matrix
    device_real_neigh = new DeviceMatrix<cufftReal>(size);
}


void run(int steps, cufftReal* device_real_grid) {
    for (int n = 0; n < steps; ++n) {
        fft->rfft2(device_real_grid, device_cplx_grid->ptr);
        scaled_hadamart_product<<<dim3(gsize1), dim3(bsize1)>>>(device_cplx_grid->ptr, device_cplx_kernel->ptr, gsize * gsize, gsize * gsize);
        fft->irfft2(device_cplx_grid->ptr, device_real_neigh->ptr);
        lenia_growth<<<dim3(gsize2), dim3(bsize2)>>>(device_real_grid, device_real_neigh->ptr, 0.1f, 0.135f, 0.015f, gsize * gsize);
    }
}


void finalize() {
    delete fft;
    delete device_cplx_grid;
    delete device_real_neigh;
    delete device_cplx_kernel;
}


