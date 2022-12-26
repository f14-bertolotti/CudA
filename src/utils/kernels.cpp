#pragma once

#include "matrix/host_matrix.cpp"

enum KernelType {game_of_life, larger_than_life, primordia, bell};


class Kernel: public HostMatrix<float>{
    
    int kernel_size, radius;
    float mu, sigma;

    public: 
        Kernel(KernelType type, int grid_size):HostMatrix(grid_size) {
            kernel_size = 0;
            switch (type) {
                case game_of_life: init_game_of_life_kernel(); break;
                case primordia: init_primordia_kernel(); break;
            }                
        }

        Kernel(KernelType type, int kernel_size, int grid_size):HostMatrix(grid_size) {
            this->kernel_size = kernel_size;
            switch (type) {
                case larger_than_life: init_larger_than_life_kernel(); break; 
                default: Kernel(type, grid_size);
            }
        }

        Kernel(KernelType type, int radius, float mu, float sigma, int grid_size):HostMatrix(grid_size) {
            this->kernel_size = radius * 2;
            this->radius      = radius;
            this->sigma       = sigma;
            this->mu          = mu;
            switch (type) {
                case bell: init_bell(); break;
                default: Kernel(type, kernel_size, grid_size);
            }
        }

    private:
        void init_game_of_life_kernel() {
            ptr[1]               = 1.0f;
            ptr[size-1]          = 1.0f;
            ptr[size]            = 1.0f;
            ptr[size+1]          = 1.0f;
            ptr[2*size-1]        = 1.0f;
            ptr[size*(size-1)]   = 1.0f;
            ptr[size*(size-1)+1] = 1.0f;
            ptr[size*size-1]     = 1.0f;
        }

        void init_larger_than_life_kernel() {

            for(int i = -kernel_size/2; i < (kernel_size%2 ? kernel_size/2+1 : kernel_size/2); ++i)
               for(int j = -kernel_size/2; j < (kernel_size%2 ? kernel_size/2+1 : kernel_size/2); ++j)
                   ptr[(i>=0?i:size+i)*size+(j>=0?j:size+j)] = 1;
        }

        void init_primordia_kernel() {
            ptr[1]               = 1.0f/8.0f;
            ptr[size-1]          = 1.0f/8.0f;
            ptr[size]            = 1.0f/8.0f;
            ptr[size+1]          = 1.0f/8.0f;
            ptr[2*size-1]        = 1.0f/8.0f;
            ptr[size*(size-1)]   = 1.0f/8.0f;
            ptr[size*(size-1)+1] = 1.0f/8.0f;
            ptr[size*size-1]     = 1.0f/8.0f;
        }

        void init_bell() {
            float sum = 0;
            for(int i = -kernel_size/2+1; i < (kernel_size%2 ? kernel_size/2+1 : kernel_size/2)+1; ++i) {
                for(int j = -kernel_size/2+1; j < (kernel_size%2 ? kernel_size/2+1 : kernel_size/2)+1; ++j) {
                    int iptr = i>=0?i:size+i;
                    int jptr = j>=0?j:size+j;
                    float norm = sqrt(i*i + j*j) / (kernel_size / 2);
                    float value = (norm < 1) * exp(-pow((norm - this->mu)/this->sigma,2.0f) / 2.0f);
                    ptr[iptr*size+jptr] = value;
                    sum += value;
               }
            }
            for (int i = 0; i < size * size; ++i) ptr[i] *= 1.0f/sum;
        }
};


