#pragma once

#include "../buffer/host_buffer.cpp"
#include "device_matrix.cpp"

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

      ~HostMatrix() {
         delete this->buffer;
      }

};
