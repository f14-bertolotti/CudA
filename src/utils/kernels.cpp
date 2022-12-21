#pragma once

#include "matrix/host_matrix.cpp"

enum KernelType {game_of_life};


class Kernel: public HostMatrix<float>{

    public: 
        Kernel(KernelType type, int kernel_size):HostMatrix(kernel_size) {
            switch (type) {
                case game_of_life: init_game_of_life_kernel();
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
};


