#pragma once

#include "matrix/host_matrix.cpp"

enum KernelType {game_of_life, larger_than_life};


class Kernel: public HostMatrix<float>{
    
    int kernel_size;

    public: 
        Kernel(KernelType type, int grid_size):HostMatrix(grid_size) {
            kernel_size = 0;
            switch (type) {
                case game_of_life: init_game_of_life_kernel();
            }                
        }

        Kernel(KernelType type, int kernel_size, int grid_size):HostMatrix(grid_size) {
            this->kernel_size = kernel_size;
            switch (type) {
                case larger_than_life: init_larger_than_life_kernel(); 
                default: Kernel(type, grid_size);
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
};


