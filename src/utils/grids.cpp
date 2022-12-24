#pragma once

#include <cstdlib>
#include <ctime>

#include "matrix/host_matrix.cpp"

enum GridType {zero, random_0_1, random_0_to_1};

class Grid : public HostMatrix<float> {

    public:
        Grid(GridType type, int grid_size) : HostMatrix(grid_size) {
            std::srand(std::time(nullptr));
            switch (type) {
                case random_0_1: init_random_0_1_grid(); break;
                case random_0_to_1: init_random_0_to_1_grid(); break;
                case zero: break;
            }
        }

    private:
        void init_random_0_1_grid() {
            for (int i = 0; i < size * size; ++i) ptr[i] = std::rand() % 2;
        }

        void init_random_0_to_1_grid() {
            for (int i = 0; i < size * size; ++i) ptr[i] = (std::rand() % 1000)/1000.0f;
        }
};
