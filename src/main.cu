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
#include <time.h>
#include "./kernels.cu"


int main(int argc, char* argv[]) {

    int size           = 1024;
    int bytes          = sizeof(int) * size * size;
    int* host_grid     = (int*) malloc(bytes);
    int* cuda_grid     = NULL;
    int* cuda_tmp_grid = NULL;
    int* cuda_new_grid = NULL;

    cudaMalloc(&cuda_grid    , bytes);
    cudaMalloc(&cuda_new_grid, bytes);


    sfRenderWindow* window = sfRenderWindow_create((sfVideoMode){800, 600, 32}, "game of life", sfResize | sfClose, NULL);
    if (!window) return EXIT_FAILURE;
    sfVertexArray* vertex_array = sfVertexArray_create();
    sfVertexArray_setPrimitiveType(vertex_array, sfPoints);
    
    srand(1);
    for (int i = 0; i < size * size; ++i) {
        host_grid[i] = rand() % 2;
        sfVertex vertex;
        vertex.color = host_grid[i] ? sfWhite : sfBlack;
        vertex.position = (sfVector2f){(float) i/size, (float) (i % size)};
        sfVertexArray_append(vertex_array, vertex);
    }

    dim3 threadsPerBlock     (16,16);
    dim3 numBlocks           (64,64);
    dim3 swapThreadsPerBlock (256);
    dim3 swapNumBlocks       (4);

    
    cudaMemcpy(cuda_grid, host_grid, bytes, cudaMemcpyHostToDevice);

    // main loop
    float avg_clock = 0;
    for(int n = 0;; ++n) {
        // start timer
        clock_t start = clock();

        swap_top_bottom<<<swapNumBlocks, swapThreadsPerBlock>>>(size, cuda_grid);
        swap_left_right<<<swapNumBlocks, swapThreadsPerBlock>>>(size, cuda_grid);
        game_of_life   <<<    numBlocks,     threadsPerBlock>>>(size, cuda_grid, cuda_new_grid);

        cuda_tmp_grid = cuda_grid;
        cuda_grid     = cuda_new_grid;
        cuda_new_grid = cuda_tmp_grid;

        for (int i = 0; i < size * size; ++i) {
            sfVertex* vertex = sfVertexArray_getVertex(vertex_array, i);
            vertex->color = host_grid[i] ? sfWhite : sfBlack;
        }

        sfRenderWindow_drawVertexArray(window, vertex_array, NULL);
        sfRenderWindow_display(window);

        cudaMemcpy(host_grid, cuda_grid, bytes, cudaMemcpyDeviceToHost);

        // take time
        int msec = ((clock() - start) * 1000 / CLOCKS_PER_SEC)%1000;
        if(n > 100) avg_clock = (msec + (n-100) * avg_clock) / (n+1-100);
        printf("\rmsec: %d, avg:%f.", msec, avg_clock);
        fflush(stdout);
    }

    // free memory
    cudaFree(cuda_grid);
    cudaFree(cuda_new_grid);
    free(host_grid);

    return 0;

}
