#include <cufft.h>

#include "../utils/window.cpp"
#include "../utils/kernels.cpp"
#include "../utils/grids.cpp"
#include "../utils/cufft.cpp"
#include "../utils/timer.cpp"
#include "../utils/utils.cu"

int main(int argc, char *argv[]) {

    int size = 512;

    FFT       fft(size);
    Grid     grid(random_0_to_1, size);
    Kernel kernel(bell, 13, 0.5, 0.15, size);
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

    for (int n = 0;; ++n) {

        timer.start();

        fft.rfft2(device_real_grid.ptr, device_cplx_grid.ptr);
        scaled_hadamart_product<<<dim3(gsize1), dim3(bsize1)>>>(device_cplx_grid.ptr, device_cplx_kernel.ptr, size * size, size * size);
        fft.irfft2(device_cplx_grid.ptr, device_real_neigh.ptr);
        lenia_growth<<<dim3(gsize2), dim3(bsize2)>>>(device_real_grid.ptr, device_real_neigh.ptr, 0.1f, 0.135f, 0.015f, size * size);
        window.update(device_real_grid.buffer);

        timer.stop();
        timer.print();

    }

    return 0;
} 

