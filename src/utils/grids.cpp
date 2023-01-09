#pragma once

#include <cstdlib>
#include <ctime>

#include "matrix/host_matrix.cpp"

enum GridType {zero, random_0_1, random_0_to_1, orbium, trained_signal};

class Grid : public HostMatrix<float> {

    public:
        Grid(GridType type, int grid_size) : HostMatrix(grid_size) {
            std::srand(std::time(nullptr));
            switch (type) {
                case random_0_1: init_random_0_1_grid(); break;
                case random_0_to_1: init_random_0_to_1_grid(); break;
                case orbium: init_orbium_grid(); break;
                case trained_signal: init_trained_signal(); break;
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

        void init_orbium_grid() {
            ptr[0*size+0]  = 0.0f  ; ptr[0*size+1] = 0.0f  ; ptr[0*size+2] = 0.0f   ; ptr[0*size+3] = 0.0f   ; ptr[0*size+4] = 0.0f  ; ptr[0*size+5] = 0.0f   ; ptr[0*size+6] = 0.1f   ; ptr[0*size+7] = 0.14f  ; ptr[0*size+8] = 0.1f   ; ptr[0*size+9] = 0.0f   ; ptr[0*size+10] = 0.0f   ; ptr[0*size+11] = 0.03f  ; ptr[0*size+12] = 0.03f  ; ptr[0*size+13] = 0.0f   ; ptr[0*size+14] = 0.0f   ; ptr[0*size+15] = 0.3f   ; ptr[0*size+16] = 0.0f   ; ptr[0*size+17] = 0.0f   ; ptr[0*size+18] = 0.0f   ; ptr[0*size+19] = 0.0f   ;
            ptr[1*size+0]  = 0.0f  ; ptr[1*size+1] = 0.0f  ; ptr[1*size+2] = 0.0f   ; ptr[1*size+3] = 0.0f   ; ptr[1*size+4] = 0.0f  ; ptr[1*size+5] = 0.08f  ; ptr[1*size+6] = 0.24f  ; ptr[1*size+7] = 0.3f   ; ptr[1*size+8] = 0.3f   ; ptr[1*size+9] = 0.18f  ; ptr[1*size+10] = 0.14f  ; ptr[1*size+11] = 0.15f  ; ptr[1*size+12] = 0.16f  ; ptr[1*size+13] = 0.15f  ; ptr[1*size+14] = 0.09f  ; ptr[1*size+15] = 0.2f   ; ptr[1*size+16] = 0.0f   ; ptr[1*size+17] = 0.0f   ; ptr[1*size+18] = 0.0f   ; ptr[1*size+19] = 0.0f   ;
            ptr[2*size+0]  = 0.0f  ; ptr[2*size+1] = 0.0f  ; ptr[2*size+2] = 0.0f   ; ptr[2*size+3] = 0.0f   ; ptr[2*size+4] = 0.0f  ; ptr[2*size+5] = 0.15f  ; ptr[2*size+6] = 0.34f  ; ptr[2*size+7] = 0.44f  ; ptr[2*size+8] = 0.46f  ; ptr[2*size+9] = 0.38f  ; ptr[2*size+10] = 0.18f  ; ptr[2*size+11] = 0.14f  ; ptr[2*size+12] = 0.11f  ; ptr[2*size+13] = 0.13f  ; ptr[2*size+14] = 0.19f  ; ptr[2*size+15] = 0.18f  ; ptr[2*size+16] = 0.45f  ; ptr[2*size+17] = 0.0f   ; ptr[2*size+18] = 0.0f   ; ptr[2*size+19] = 0.0f   ;
            ptr[3*size+0]  = 0.0f  ; ptr[3*size+1] = 0.0f  ; ptr[3*size+2] = 0.0f   ; ptr[3*size+3] = 0.0f   ; ptr[3*size+4] = 0.06f ; ptr[3*size+5] = 0.13f  ; ptr[3*size+6] = 0.39f  ; ptr[3*size+7] = 0.5f   ; ptr[3*size+8] = 0.5f   ; ptr[3*size+9] = 0.37f  ; ptr[3*size+10] = 0.06f  ; ptr[3*size+11] = 0.0f   ; ptr[3*size+12] = 0.0f   ; ptr[3*size+13] = 0.0f   ; ptr[3*size+14] = 0.02f  ; ptr[3*size+15] = 0.16f  ; ptr[3*size+16] = 0.68f  ; ptr[3*size+17] = 0.0f   ; ptr[3*size+18] = 0.0f   ; ptr[3*size+19] = 0.0f   ;
            ptr[4*size+0]  = 0.0f  ; ptr[4*size+1] = 0.0f  ; ptr[4*size+2] = 0.0f   ; ptr[4*size+3] = 0.11f  ; ptr[4*size+4] = 0.17f ; ptr[4*size+5] = 0.17f  ; ptr[4*size+6] = 0.33f  ; ptr[4*size+7] = 0.4f   ; ptr[4*size+8] = 0.38f  ; ptr[4*size+9] = 0.28f  ; ptr[4*size+10] = 0.14f  ; ptr[4*size+11] = 0.0f   ; ptr[4*size+12] = 0.0f   ; ptr[4*size+13] = 0.0f   ; ptr[4*size+14] = 0.0f   ; ptr[4*size+15] = 0.0f   ; ptr[4*size+16] = 0.18f  ; ptr[4*size+17] = 0.42f  ; ptr[4*size+18] = 0.0f   ; ptr[4*size+19] = 0.0f   ;
            ptr[5*size+0]  = 0.0f  ; ptr[5*size+1] = 0.0f  ; ptr[5*size+2] = 0.09f  ; ptr[5*size+3] = 0.18f  ; ptr[5*size+4] = 0.13f ; ptr[5*size+5] = 0.06f  ; ptr[5*size+6] = 0.08f  ; ptr[5*size+7] = 0.26f  ; ptr[5*size+8] = 0.32f  ; ptr[5*size+9] = 0.32f  ; ptr[5*size+10] = 0.27f  ; ptr[5*size+11] = 0.0f   ; ptr[5*size+12] = 0.0f   ; ptr[5*size+13] = 0.0f   ; ptr[5*size+14] = 0.0f   ; ptr[5*size+15] = 0.0f   ; ptr[5*size+16] = 0.0f   ; ptr[5*size+17] = 0.82f  ; ptr[5*size+18] = 0.0f   ; ptr[5*size+19] = 0.0f   ;
            ptr[6*size+0]  = 0.27f ; ptr[6*size+1] = 0.0f  ; ptr[6*size+2] = 0.16f  ; ptr[6*size+3] = 0.12f  ; ptr[6*size+4] = 0.0f  ; ptr[6*size+5] = 0.0f   ; ptr[6*size+6] = 0.0f   ; ptr[6*size+7] = 0.25f  ; ptr[6*size+8] = 0.38f  ; ptr[6*size+9] = 0.44f  ; ptr[6*size+10] = 0.45f  ; ptr[6*size+11] = 0.34f  ; ptr[6*size+12] = 0.0f   ; ptr[6*size+13] = 0.0f   ; ptr[6*size+14] = 0.0f   ; ptr[6*size+15] = 0.0f   ; ptr[6*size+16] = 0.0f   ; ptr[6*size+17] = 0.22f  ; ptr[6*size+18] = 0.17f  ; ptr[6*size+19] = 0.0f   ;
            ptr[7*size+0]  = 0.0f  ; ptr[7*size+1] = 0.07f ; ptr[7*size+2] = 0.2f   ; ptr[7*size+3] = 0.02f  ; ptr[7*size+4] = 0.0f  ; ptr[7*size+5] = 0.0f   ; ptr[7*size+6] = 0.0f   ; ptr[7*size+7] = 0.31f  ; ptr[7*size+8] = 0.48f  ; ptr[7*size+9] = 0.57f  ; ptr[7*size+10] = 0.6f   ; ptr[7*size+11] = 0.57f  ; ptr[7*size+12] = 0.0f   ; ptr[7*size+13] = 0.0f   ; ptr[7*size+14] = 0.0f   ; ptr[7*size+15] = 0.0f   ; ptr[7*size+16] = 0.0f   ; ptr[7*size+17] = 0.0f   ; ptr[7*size+18] = 0.49f  ; ptr[7*size+19] = 0.0f   ;
            ptr[8*size+0]  = 0.0f  ; ptr[8*size+1] = 0.59f ; ptr[8*size+2] = 0.19f  ; ptr[8*size+3] = 0.0f   ; ptr[8*size+4] = 0.0f  ; ptr[8*size+5] = 0.0f   ; ptr[8*size+6] = 0.0f   ; ptr[8*size+7] = 0.2f   ; ptr[8*size+8] = 0.57f  ; ptr[8*size+9] = 0.69f  ; ptr[8*size+10] = 0.76f  ; ptr[8*size+11] = 0.76f  ; ptr[8*size+12] = 0.49f  ; ptr[8*size+13] = 0.0f   ; ptr[8*size+14] = 0.0f   ; ptr[8*size+15] = 0.0f   ; ptr[8*size+16] = 0.0f   ; ptr[8*size+17] = 0.0f   ; ptr[8*size+18] = 0.36f  ; ptr[8*size+19] = 0.0f   ;
            ptr[9*size+0]  = 0.0f  ; ptr[9*size+1] = 0.58f ; ptr[9*size+2] = 0.19f  ; ptr[9*size+3] = 0.0f   ; ptr[9*size+4] = 0.0f  ; ptr[9*size+5] = 0.0f   ; ptr[9*size+6] = 0.0f   ; ptr[9*size+7] = 0.0f   ; ptr[9*size+8] = 0.67f  ; ptr[9*size+9] = 0.83f  ; ptr[9*size+10] = 0.9f   ; ptr[9*size+11] = 0.92f  ; ptr[9*size+12] = 0.87f  ; ptr[9*size+13] = 0.12f  ; ptr[9*size+14] = 0.0f   ; ptr[9*size+15] = 0.0f   ; ptr[9*size+16] = 0.0f   ; ptr[9*size+17] = 0.0f   ; ptr[9*size+18] = 0.22f  ; ptr[9*size+19] = 0.07f  ;
            ptr[10*size+0] = 0.0f  ; ptr[10*size+1] = 0.0f ; ptr[10*size+2] = 0.46f ; ptr[10*size+3] = 0.0f  ; ptr[10*size+4] = 0.0f ; ptr[10*size+5] = 0.0f  ; ptr[10*size+6] = 0.0f  ; ptr[10*size+7] = 0.0f  ; ptr[10*size+8] = 0.7f  ; ptr[10*size+9] = 0.93f ; ptr[10*size+10] = 1.0f  ; ptr[10*size+11] = 1.0f  ; ptr[10*size+12] = 1.0f  ; ptr[10*size+13] = 0.61f ; ptr[10*size+14] = 0.0f  ; ptr[10*size+15] = 0.0f  ; ptr[10*size+16] = 0.0f  ; ptr[10*size+17] = 0.0f  ; ptr[10*size+18] = 0.18f ; ptr[10*size+19] = 0.11f ;
            ptr[11*size+0] = 0.0f  ; ptr[11*size+1] = 0.0f ; ptr[11*size+2] = 0.82f ; ptr[11*size+3] = 0.0f  ; ptr[11*size+4] = 0.0f ; ptr[11*size+5] = 0.0f  ; ptr[11*size+6] = 0.0f  ; ptr[11*size+7] = 0.0f  ; ptr[11*size+8] = 0.47f ; ptr[11*size+9] = 1.0f  ; ptr[11*size+10] = 1.0f  ; ptr[11*size+11] = 0.98f ; ptr[11*size+12] = 1.0f  ; ptr[11*size+13] = 0.96f ; ptr[11*size+14] = 0.27f ; ptr[11*size+15] = 0.0f  ; ptr[11*size+16] = 0.0f  ; ptr[11*size+17] = 0.0f  ; ptr[11*size+18] = 0.19f ; ptr[11*size+19] = 0.1f  ;
            ptr[12*size+0] = 0.0f  ; ptr[12*size+1] = 0.0f ; ptr[12*size+2] = 0.46f ; ptr[12*size+3] = 0.0f  ; ptr[12*size+4] = 0.0f ; ptr[12*size+5] = 0.0f  ; ptr[12*size+6] = 0.0f  ; ptr[12*size+7] = 0.0f  ; ptr[12*size+8] = 0.25f ; ptr[12*size+9] = 1.0f  ; ptr[12*size+10] = 1.0f  ; ptr[12*size+11] = 0.84f ; ptr[12*size+12] = 0.92f ; ptr[12*size+13] = 0.97f ; ptr[12*size+14] = 0.54f ; ptr[12*size+15] = 0.14f ; ptr[12*size+16] = 0.04f ; ptr[12*size+17] = 0.1f  ; ptr[12*size+18] = 0.21f ; ptr[12*size+19] = 0.05f ;
            ptr[13*size+0] = 0.0f  ; ptr[13*size+1] = 0.0f ; ptr[13*size+2] = 0.0f  ; ptr[13*size+3] = 0.4f  ; ptr[13*size+4] = 0.0f ; ptr[13*size+5] = 0.0f  ; ptr[13*size+6] = 0.0f  ; ptr[13*size+7] = 0.0f  ; ptr[13*size+8] = 0.09f ; ptr[13*size+9] = 0.8f  ; ptr[13*size+10] = 1.0f  ; ptr[13*size+11] = 0.82f ; ptr[13*size+12] = 0.8f  ; ptr[13*size+13] = 0.85f ; ptr[13*size+14] = 0.63f ; ptr[13*size+15] = 0.31f ; ptr[13*size+16] = 0.18f ; ptr[13*size+17] = 0.19f ; ptr[13*size+18] = 0.2f  ; ptr[13*size+19] = 0.01f ;
            ptr[14*size+0] = 0.0f  ; ptr[14*size+1] = 0.0f ; ptr[14*size+2] = 0.0f  ; ptr[14*size+3] = 0.36f ; ptr[14*size+4] = 0.1f ; ptr[14*size+5] = 0.0f  ; ptr[14*size+6] = 0.0f  ; ptr[14*size+7] = 0.0f  ; ptr[14*size+8] = 0.05f ; ptr[14*size+9] = 0.54f ; ptr[14*size+10] = 0.86f ; ptr[14*size+11] = 0.79f ; ptr[14*size+12] = 0.74f ; ptr[14*size+13] = 0.72f ; ptr[14*size+14] = 0.6f  ; ptr[14*size+15] = 0.39f ; ptr[14*size+16] = 0.28f ; ptr[14*size+17] = 0.24f ; ptr[14*size+18] = 0.13f ; ptr[14*size+19] = 0.0f  ;
            ptr[15*size+0] = 0.0f  ; ptr[15*size+1] = 0.0f ; ptr[15*size+2] = 0.0f  ; ptr[15*size+3] = 0.01f ; ptr[15*size+4] = 0.3f ; ptr[15*size+5] = 0.07f ; ptr[15*size+6] = 0.0f  ; ptr[15*size+7] = 0.0f  ; ptr[15*size+8] = 0.08f ; ptr[15*size+9] = 0.36f ; ptr[15*size+10] = 0.64f ; ptr[15*size+11] = 0.7f  ; ptr[15*size+12] = 0.64f ; ptr[15*size+13] = 0.6f  ; ptr[15*size+14] = 0.51f ; ptr[15*size+15] = 0.39f ; ptr[15*size+16] = 0.29f ; ptr[15*size+17] = 0.19f ; ptr[15*size+18] = 0.04f ; ptr[15*size+19] = 0.0f  ;
            ptr[16*size+0] = 0.0f  ; ptr[16*size+1] = 0.0f ; ptr[16*size+2] = 0.0f  ; ptr[16*size+3] = 0.0f  ; ptr[16*size+4] = 0.1f ; ptr[16*size+5] = 0.24f ; ptr[16*size+6] = 0.14f ; ptr[16*size+7] = 0.1f  ; ptr[16*size+8] = 0.15f ; ptr[16*size+9] = 0.29f ; ptr[16*size+10] = 0.45f ; ptr[16*size+11] = 0.53f ; ptr[16*size+12] = 0.52f ; ptr[16*size+13] = 0.46f ; ptr[16*size+14] = 0.4f  ; ptr[16*size+15] = 0.31f ; ptr[16*size+16] = 0.21f ; ptr[16*size+17] = 0.08f ; ptr[16*size+18] = 0.0f  ; ptr[16*size+19] = 0.0f  ;
            ptr[17*size+0] = 0.0f  ; ptr[17*size+1] = 0.0f ; ptr[17*size+2] = 0.0f  ; ptr[17*size+3] = 0.0f  ; ptr[17*size+4] = 0.0f ; ptr[17*size+5] = 0.08f ; ptr[17*size+6] = 0.21f ; ptr[17*size+7] = 0.21f ; ptr[17*size+8] = 0.22f ; ptr[17*size+9] = 0.29f ; ptr[17*size+10] = 0.36f ; ptr[17*size+11] = 0.39f ; ptr[17*size+12] = 0.37f ; ptr[17*size+13] = 0.33f ; ptr[17*size+14] = 0.26f ; ptr[17*size+15] = 0.18f ; ptr[17*size+16] = 0.09f ; ptr[17*size+17] = 0.0f  ; ptr[17*size+18] = 0.0f  ; ptr[17*size+19] = 0.0f  ;
            ptr[18*size+0] = 0.0f  ; ptr[18*size+1] = 0.0f ; ptr[18*size+2] = 0.0f  ; ptr[18*size+3] = 0.0f  ; ptr[18*size+4] = 0.0f ; ptr[18*size+5] = 0.0f  ; ptr[18*size+6] = 0.03f ; ptr[18*size+7] = 0.13f ; ptr[18*size+8] = 0.19f ; ptr[18*size+9] = 0.22f ; ptr[18*size+10] = 0.24f ; ptr[18*size+11] = 0.24f ; ptr[18*size+12] = 0.23f ; ptr[18*size+13] = 0.18f ; ptr[18*size+14] = 0.13f ; ptr[18*size+15] = 0.05f ; ptr[18*size+16] = 0.0f  ; ptr[18*size+17] = 0.0f  ; ptr[18*size+18] = 0.0f  ; ptr[18*size+19] = 0.0f  ;
            ptr[19*size+0] = 0.0f  ; ptr[19*size+1] = 0.0f ; ptr[19*size+2] = 0.0f  ; ptr[19*size+3] = 0.0f  ; ptr[19*size+4] = 0.0f ; ptr[19*size+5] = 0.0f  ; ptr[19*size+6] = 0.0f  ; ptr[19*size+7] = 0.0f  ; ptr[19*size+8] = 0.02f ; ptr[19*size+9] = 0.06f ; ptr[19*size+10] = 0.08f ; ptr[19*size+11] = 0.09f ; ptr[19*size+12] = 0.07f ; ptr[19*size+13] = 0.05f ; ptr[19*size+14] = 0.01f ; ptr[19*size+15] = 0.0f  ; ptr[19*size+16] = 0.0f  ; ptr[19*size+17] = 0.0f  ; ptr[19*size+18] = 0.0f  ; ptr[19*size+19] = 0.0f  ;
        }

        void init_trained_signal() {
ptr[0*size+1]=1.000f;ptr[0*size+4]=1.000f;ptr[0*size+7]=0.036f;ptr[0*size+8]=1.000f;ptr[0*size+15]=1.000f;ptr[0*size+17]=1.000f;ptr[0*size+19]=1.000f;ptr[1*size+0]=1.000f;ptr[1*size+1]=1.000f;ptr[1*size+8]=1.000f;ptr[2*size+1]=1.000f;ptr[2*size+6]=1.000f;ptr[2*size+10]=1.000f;ptr[2*size+13]=0.303f;ptr[2*size+14]=1.000f;ptr[2*size+18]=1.000f;ptr[3*size+7]=0.025f;ptr[3*size+8]=1.000f;ptr[3*size+11]=1.000f;ptr[3*size+14]=1.000f;ptr[3*size+15]=1.000f;ptr[4*size+13]=1.000f;ptr[4*size+15]=1.000f;ptr[5*size+2]=1.000f;ptr[5*size+3]=1.000f;ptr[5*size+5]=1.000f;ptr[5*size+7]=1.000f;ptr[5*size+12]=1.000f;ptr[5*size+14]=1.000f;ptr[5*size+16]=1.000f;ptr[5*size+19]=1.000f;ptr[6*size+2]=0.077f;ptr[6*size+3]=1.000f;ptr[6*size+6]=1.000f;ptr[6*size+9]=1.000f;ptr[6*size+14]=1.000f;ptr[7*size+3]=1.000f;ptr[7*size+5]=0.110f;ptr[7*size+17]=1.000f;ptr[7*size+18]=1.000f;ptr[8*size+8]=1.000f;ptr[8*size+10]=1.000f;ptr[8*size+11]=1.000f;ptr[8*size+15]=0.136f;ptr[8*size+18]=1.000f;ptr[9*size+4]=1.000f;ptr[9*size+5]=1.000f;ptr[9*size+7]=1.000f;ptr[9*size+10]=1.000f;ptr[9*size+13]=1.000f;ptr[9*size+15]=1.000f;ptr[9*size+16]=1.000f;ptr[9*size+17]=1.000f;ptr[9*size+19]=1.000f;ptr[10*size+10]=1.000f;ptr[10*size+19]=1.000f;ptr[11*size+4]=1.000f;ptr[11*size+5]=1.000f;ptr[12*size+4]=1.000f;ptr[12*size+5]=1.000f;ptr[12*size+8]=1.000f;ptr[12*size+12]=1.000f;ptr[12*size+17]=1.000f;ptr[13*size+0]=1.000f;ptr[13*size+6]=1.000f;ptr[13*size+10]=1.000f;ptr[13*size+13]=1.000f;ptr[14*size+9]=1.000f;ptr[14*size+17]=1.000f;ptr[14*size+19]=1.000f;ptr[15*size+8]=1.000f;ptr[15*size+9]=1.000f;ptr[15*size+11]=1.000f;ptr[15*size+12]=1.000f;ptr[15*size+13]=1.000f;ptr[15*size+15]=1.000f;ptr[16*size+15]=1.000f;ptr[18*size+3]=1.000f;ptr[18*size+14]=1.000f;ptr[19*size+1]=1.000f;ptr[19*size+10]=1.000f;ptr[19*size+18]=1.000f;

}
};
