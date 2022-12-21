#pragma once

#include "../buffer/device_buffer.cpp"

template <class T>
class HostMatrix;

template <class T>
class DeviceMatrix {
    
    public:
        DeviceBuffer<T>* buffer;
        T* ptr;
        int size;

        DeviceMatrix(int size) {
            buffer     = new DeviceBuffer<T>(size * size);
            this->ptr  = buffer->ptr;
            this->size = size;
        }

        DeviceMatrix(HostMatrix<T>* matrix) {
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

        ~DeviceMatrix() { 
            delete this->buffer;
        }
        
};
