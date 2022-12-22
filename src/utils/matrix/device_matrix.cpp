#pragma once

#include "../buffer/device_buffer.cpp"
#include <cufft.h>

template <class T>
class HostMatrix;

template <class T>
__global__ void hadamart_product(T* data, float scalar, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) data[i] *= scalar;
}

template <>
__global__ void hadamart_product<cufftComplex>(cufftComplex* data, float scalar, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) data[i].x *= scalar;
}




template <class T>
class DeviceMatrix {

    private:
        int hadamart_product_grid_size, hadamart_product_block_size;
    
    public:
        DeviceBuffer<T>* buffer;
        T* ptr;
        int size;

        DeviceMatrix(int size) {
            cudaOccupancyMaxPotentialBlockSize(&hadamart_product_block_size, &hadamart_product_block_size, hadamart_product<T>, 0, size * size);
            buffer     = new DeviceBuffer<T>(size * size);
            this->ptr  = buffer->ptr;
            this->size = size;
        }

        DeviceMatrix(HostMatrix<T>* matrix) {
            cudaOccupancyMaxPotentialBlockSize(&hadamart_product_block_size, &hadamart_product_block_size, hadamart_product<T>, 0, size * size);
            this->size = matrix->size;
            this->buffer = new DeviceBuffer<T>(matrix->buffer);
            this->ptr = this->buffer->ptr;
        }

        void print(const char* format) {
            T* tmp = (T*) std::calloc(size * size, sizeof(T));
            cudaMemcpy(tmp, ptr, size * size * sizeof(T), cudaMemcpyDeviceToHost);
            for (int i = 1; i < size * size + 1; ++i) {
                printf(format, tmp[i-1]);
                if (i % size == 0) printf("\n"); 
            }
            free(tmp);
        }

        DeviceMatrix* scale(float value) {
            hadamart_product<T><<<dim3(hadamart_product_grid_size),dim3(hadamart_product_block_size)>>>(this->ptr, value, size*size);
        }

        ~DeviceMatrix() { 
            delete this->buffer;
        }
        
};
