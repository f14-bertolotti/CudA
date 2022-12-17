#pragma once

#include <cstdlib>
#include <ctime>

#include "matrix.cpp"

enum GridType {zero, random_0_1};

class Grid : public Matrix<float> {

    public:
        Grid(GridType type, int grid_size) : Matrix(grid_size) {
            switch (type) {
                case random_0_1: init_random_0_1_grid(); break;
                case zero: break;
            }
        }

    private:
        void init_random_0_1_grid() {
            std::srand(std::time(nullptr));
            for (int i = 0; i < size * size; ++i) ptr[i] = std::rand() % 1000 / 1000.0f;
        }
};
