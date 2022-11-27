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

#define BLOCK_SIZE 16


__global__ void game_of_life(int size, int* grid, int* new_grid) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int id = iy * size + ix;
    int numNeighbors;

    if (1 < ix && ix < size-1 && 1 < iy && iy < size-1) {
        numNeighbors = grid[(ix-1) * size + iy+1] + grid[(ix+0) * size + iy+1] + grid[(ix+1) * size + iy+1] + 
                       grid[(ix-1) * size + iy+0] + grid[(ix+0) * size + iy+0] + grid[(ix+1) * size + iy+0] +
                       grid[(ix-1) * size + iy-1] + grid[(ix+0) * size + iy-1] + grid[(ix+1) * size + iy-1]; 
        new_grid[id] = 0 * (new_grid[id] == 1 && numNeighbors < 2) + 
                       1 * (new_grid[id] == 1 && numNeighbors == 2 || numNeighbors == 3) + 
                       0 * (new_grid[id] == 1 && numNeighbors > 3) +
                       1 * (new_grid[id] == 0 && numNeighbors == 3) + 
                       new_grid[id] * (new_grid[id] == 0 && numNeighbors != 3);
    }
}

__global__ void GOL(int dim, int *grid, int *newGrid)
{
    // We want id ∈ [1,dim]
    int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int id = iy * (dim+2) + ix;
    
    int numNeighbors;
    
    if (iy < dim-2 && ix < dim-2) {
    
        // Get the number of neighbors for a given grid point
        numNeighbors = grid[id+(dim+2)] + grid[id-(dim+2)] //upper lower
                    + grid[id+1] + grid[id-1]             //right left
                    + grid[id+(dim+3)] + grid[id-(dim+3)] //diagonals
                    + grid[id-(dim+1)] + grid[id+(dim+1)];
        
        int cell = grid[id];
        // Here we have explicitly all of the game rules
        if (cell == 1 && numNeighbors < 2)
            newGrid[id] = 0;
        else if (cell == 1 && (numNeighbors == 2 || numNeighbors == 3))
            newGrid[id] = 1;
        else if (cell == 1 && numNeighbors > 3)
            newGrid[id] = 0;
        else if (cell == 0 && numNeighbors == 3)
            newGrid[id] = 1;
        else
            newGrid[id] = cell;
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
            host_grid[i * size + j] = rand() % 2;
            sfVertex vertex;
            vertex.color = host_grid[i * size + j] ? sfWhite : sfBlack;
            vertex.position = (sfVector2f){(float) i, (float) j};
            sfVertexArray_append(vertex_array, vertex);
        }
    }


    cudaMemcpy(cuda_grid, host_grid, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(128,128);

    sfUint8* update = (sfUint8*) malloc(sizeof(sfUint8) * size * size);
    while (sfRenderWindow_isOpen(window)) {
        game_of_life<<<numBlocks,threadsPerBlock>>>(size, cuda_grid, cuda_new_grid);
        cuda_tmp_grid = cuda_grid;
        cuda_grid = cuda_new_grid;
        cuda_new_grid = cuda_tmp_grid;       
        cudaMemcpy(host_grid, cuda_grid, bytes, cudaMemcpyDeviceToHost);


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
