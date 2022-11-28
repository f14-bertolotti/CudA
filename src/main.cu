#include <SFML/Config.h>
#include <SFML/Graphics/Color.h>
#include <SFML/Graphics/Image.h>
#include <SFML/Graphics/PrimitiveType.h>
#include <SFML/Graphics/Rect.h>
#include <SFML/Graphics/RenderTexture.h>
#include <SFML/Graphics/RenderWindow.h>
#include <SFML/Graphics/Sprite.h>
#include <SFML/Graphics/Text.h>
#include <SFML/Graphics/Texture.h>
#include <SFML/Graphics/Types.h>
#include <SFML/Graphics/Vertex.h>
#include <SFML/Graphics/VertexArray.h>
#include <SFML/Graphics/VertexBuffer.h>
#include <SFML/System/Vector2.h>
#include <SFML/Window/VideoMode.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <SFML/Graphics.h>


void print_matrix(int size, int* grid) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%d",grid[i * size + j]);
        }
        printf("\n");
    }
}

__global__ void game_of_life(int size, int* grid, int* new_grid) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int id = iy * size + ix;
    int numNeighbors;
    

    if (0 < ix && ix < size-1 && 0 < iy && iy < size-1) {
        numNeighbors = grid[(iy-1) * size + ix+1] + grid[(iy+0) * size + ix+1] + grid[(iy+1) * size + ix+1] + 
                       grid[(iy-1) * size + ix+0] + 0                          + grid[(iy+1) * size + ix+0] +
                       grid[(iy-1) * size + ix-1] + grid[(iy+0) * size + ix-1] + grid[(iy+1) * size + ix-1]; 
        new_grid[id] = 0        * (grid[id] == 1 && numNeighbors  < 2) + 
                       1        * (grid[id] == 1 && (numNeighbors == 2 || numNeighbors == 3)) + 
                       0        * (grid[id] == 1 && numNeighbors  > 3) +
                       1        * (grid[id] == 0 && numNeighbors == 3) + 
                       grid[id] * (grid[id] == 0 && numNeighbors != 3);
    }

}

int main(int argc, char* argv[]) {
    int size = 1024;

    sfRenderWindow* window = sfRenderWindow_create((sfVideoMode){800, 600, 32}, "game of life", sfResize | sfClose, NULL);
    if (!window) return EXIT_FAILURE;

    sfVertexArray* vertex_array = sfVertexArray_create();
    sfVertexArray_setPrimitiveType(vertex_array, sfPoints);
    
    int bytes = sizeof(int) * size * size;

    int* host_grid = NULL;
    int* cuda_grid = NULL;
    int* cuda_tmp_grid = NULL;
    int* cuda_new_grid = NULL;

    host_grid = (int*) malloc(bytes);
    cudaMalloc(&cuda_grid, bytes);
    cudaMalloc(&cuda_new_grid, bytes);

    srand(14);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (0 < i && i < size-1 && 0 < j && j < size-1) host_grid[i * size + j] = rand() % 2;
            sfVertex vertex;
            vertex.color = host_grid[i * size + j] ? sfWhite : sfBlack;
            vertex.position = (sfVector2f){(float) i, (float) j};
            sfVertexArray_append(vertex_array, vertex);
        }
    }
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(128,128);

    print_matrix(size, host_grid);
    printf("\n");
    cudaMemcpy(cuda_grid, host_grid, bytes, cudaMemcpyHostToDevice);


    int i = 0; 
    while (++i < 1000000) {
        printf("%d\n",i);
        game_of_life<<<numBlocks,threadsPerBlock>>>(size, cuda_grid, cuda_new_grid);
        cuda_tmp_grid = cuda_grid;
        cuda_grid = cuda_new_grid;
        cuda_new_grid = cuda_tmp_grid;       
        cudaMemcpy(host_grid, cuda_grid, bytes, cudaMemcpyDeviceToHost);

        //print_matrix(size, host_grid);


        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                sfVertex* vertex = sfVertexArray_getVertex(vertex_array, i * size + j);
                vertex->color = host_grid[i * size + j] ? sfWhite : sfBlack;
            }
        }

        sfRenderWindow_drawVertexArray(window, vertex_array, NULL);
        sfRenderWindow_display(window);
    }

    cudaFree(cuda_grid);
    cudaFree(cuda_new_grid);
    free(host_grid);

    return 0;

}
