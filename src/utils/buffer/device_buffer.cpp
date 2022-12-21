#pragma once


template <class T>
class HostBuffer;

template <class T>
class DeviceBuffer {

    public:
        T* ptr;
        int size;
        
        DeviceBuffer(int size) {
            cudaMalloc(&ptr, sizeof(T) * size);
            this->size = size;
        }

        DeviceBuffer(HostBuffer<T>* buffer) {
            this->size = buffer->size;
            cudaMalloc(&ptr, sizeof(T) * this->size);
            cudaMemcpy(ptr, buffer->ptr, sizeof(T) * this->size, cudaMemcpyHostToDevice);
        }

        void print(const char* format) {
            HostBuffer<T> buffer(this);
            buffer.print(format);
        }

        ~DeviceBuffer() {
            cudaFree(this->ptr);
        }

};
