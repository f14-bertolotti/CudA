#pragma once

template <class T>
class Matrix {

   public:
      T* ptr;
      int size;
    
      Matrix(int size) {
         this->ptr  = (T*) calloc(size * size, sizeof(T));
         this->size = size;
      }

      void print() { print("%.2f "); }

      void print(const char* format_string) {
          for (int i = 1; i < size * size + 1; ++i) {
              printf(format_string, ptr[i-1]);
              if (i % size == 0) printf("\n");
          }
      }

      Matrix<T>* dot(float a) {
         for (int i = 0; i < size * size + 1; ++i) ptr[i] *= a;
         return this;
      }

      ~Matrix() {
         free(this->ptr);
      }

};
