#pragma once
#include "matrix.cpp"

template <class T>
class DeviceMatrix {
    public:
        T* ptr;
        int size;

        DeviceMatrix(Matrix<T>* matrix) {
            size = matrix->size;
            cudaMalloc(&ptr, matrix->size * matrix->size * sizeof(T));
            cudaMemcpy(ptr, matrix->ptr, matrix->size * matrix->size * sizeof(T), cudaMemcpyHostToDevice);
        }

        DeviceMatrix(int size) {
            cudaMalloc(&ptr, size * size * sizeof(T));
            cudaMemset(ptr, 0, size * size * sizeof(T));
        }

        void print() {
            print("%.2f ");
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

        ~DeviceMatrix() { cudaFree(ptr); }
};
