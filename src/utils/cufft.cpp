#pragma once

#include <cufft.h>
#include "device_matrix.cpp"

class FFT {

    public:
        cufftHandle planR2C;
        cufftHandle planC2R;

    FFT(int size) {
        cufftPlan2d(&planR2C, size, size, CUFFT_R2C);
        cufftPlan2d(&planC2R, size, size, CUFFT_C2R);
    }

    void rfft2(DeviceMatrix<cufftReal>* input, DeviceMatrix<cufftComplex>* output) {
        cufftExecR2C(planR2C, input->ptr, output->ptr);
    }

    void irfft2(DeviceMatrix<cufftComplex>* input, DeviceMatrix<cufftReal>* output) {
        cufftExecC2R(planC2R, input->ptr, output->ptr);
    }

    ~FFT() {
        cufftDestroy(planC2R);
        cufftDestroy(planR2C);
    }

};