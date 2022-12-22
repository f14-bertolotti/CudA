#pragma once

#include "../buffer/host_buffer.cpp"
#include "device_matrix.cpp"
#include <cufft.h>

template <class T>
class HostMatrix {

   public:
      HostBuffer<T>* buffer;
      T* ptr;
      int size;
    
      HostMatrix(int size) {
         this->size   = size;
         this->buffer = new HostBuffer<T>(size * size);
         this->ptr    = this->buffer->ptr;
      }

      HostMatrix(DeviceMatrix<T>* deviceMatrix) {
         this->size = deviceMatrix->size;
         this->buffer = new HostBuffer<T>(deviceMatrix->buffer);
         this->ptr = this->buffer->ptr;
      }


      void print(const char* format_string) {
          for (int i = 1; i < size * size + 1; ++i) {
              printf(format_string, buffer->ptr[i-1]);
              if (i % size == 0) printf("\n");
          }
      }

      HostMatrix<T>* scale(float value) {
         for (int i = 0; i < size * size; ++i) buffer->ptr[i] *= value;
         return this;
      }

      ~HostMatrix() {
         delete this->buffer;
      }

};

template<> HostMatrix<cufftComplex>* HostMatrix<cufftComplex>::scale(float value) {
   for (int i = 0; i < size * size; ++i) buffer->ptr[i].x *= value;
   return this;
}
