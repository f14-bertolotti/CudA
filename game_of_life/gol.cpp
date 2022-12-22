#include <cufft.h>

#include "../src/utils/window.cpp"
#include "../src/utils/kernels.cpp"
#include "../src/utils/grids.cpp"
#include "../src/utils/cufft.cpp"
#include "../src/utils/timer.cpp"
#include "../src/utils/utils.cu"

int main(int argc, char *argv[]) {

    int size = 1024;

    FFT       fft(size);
    Grid     grid(  random_0_1, size);
    Kernel kernel(game_of_life, size);
    Window window(size);
    Timer   timer;

    DeviceMatrix<cufftReal>    device_real_grid(&grid);
    DeviceMatrix<cufftComplex> device_cplx_grid(size);
    DeviceMatrix<cufftReal>    device_real_neigh(size);
    DeviceMatrix<cufftComplex> device_cplx_neigh(size);
    DeviceMatrix<cufftReal>    device_real_kernel(&kernel);
    DeviceMatrix<cufftComplex> device_cplx_kernel(size);

    fft.rfft2(&device_real_kernel, &device_cplx_kernel);

    int gsize1, bsize1, gsize2, bsize2;
    cudaOccupancyMaxPotentialBlockSize(&gsize1, &bsize1, scaled_hadamart_product, 0, size * size);
    cudaOccupancyMaxPotentialBlockSize(&gsize2, &bsize2,     game_of_life_growth, 0, size * size);
    gsize1 = ((size*size) + bsize1 - 1) / bsize1; 
    gsize2 = ((size*size) + bsize2 - 1) / bsize2; 

    for (int n = 0; ; ++n) {

        timer.start();
 
        fft.rfft2(&device_real_grid, &device_cplx_grid);
        scaled_hadamart_product<<<dim3(gsize1), dim3(bsize1)>>>(device_cplx_grid.ptr, device_cplx_kernel.ptr, size * size, size * size);
        fft.irfft2(&device_cplx_grid, &device_real_neigh);
        game_of_life_growth<<<dim3(gsize2), dim3(bsize2)>>>(device_real_grid.ptr, device_real_neigh.ptr, size * size);
        window.update(device_real_grid.buffer);
        
        timer.stop();
        timer.print();

    }

    return 0;
} 

