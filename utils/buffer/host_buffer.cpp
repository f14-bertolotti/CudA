#pragma once

#include "device_buffer.cpp"

template <class T>
class HostBuffer {

    public:
        T* ptr;
        int size;
        
        HostBuffer(int size) {
           this->size = size;
           this->ptr  = (T*) std::calloc(size, sizeof(T));
        }

        HostBuffer(DeviceBuffer<T> buffer) {
            this->size = buffer->size;
            this->ptr = (T*) std::malloc(size * sizeof(T));
            cudaMemcpy(ptr, buffer->ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
        }

        void print(const char* format) {
            for (int i = 0; i < size; ++i) printf(format, this->ptr[i]);
        }

        ~HostBuffer() {
            free(this->ptr);
        }

};
