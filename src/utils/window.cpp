#pragma once

#include <SFML/Graphics.hpp>
#include "buffer/device_buffer.cpp"
#include "buffer/host_buffer.cpp"


__global__ void copy_from(sf::Uint8* color_buffer, float* data_buffer, unsigned int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        color_buffer[4 * i + 0] = min(data_buffer[i] * 255+30, 255.0f);
	    color_buffer[4 * i + 1] = min(data_buffer[i] * 255+30, 255.0f);
	    color_buffer[4 * i + 2] = min(data_buffer[i] * 255+30, 255.0f);
	    color_buffer[4 * i + 3] = 255;
    }
}

class Window {

    public:
        int size;
        sf::RenderWindow* window;
        sf::Texture      texture;
        sf::Sprite        sprite;

        HostBuffer  <sf::Uint8>*   host_buffer;
        DeviceBuffer<sf::Uint8>* device_buffer;

        Window(int size) {
            this->size   = size;
            this->window = new sf::RenderWindow(sf::VideoMode(size, size), "cellular automaton in cuda");

	        this->  host_buffer = new   HostBuffer<sf::Uint8>(size * size * 4);
            this->device_buffer = new DeviceBuffer<sf::Uint8>(size * size * 4);

	        texture.create(size, size);
        }

        void update(DeviceBuffer<float>* buffer) {
            
            int gsize, bsize;
            cudaOccupancyMaxPotentialBlockSize(&gsize, &bsize, copy_from, 0, 0);
            gsize = ((size*size) + bsize - 1) / bsize; 
    
            copy_from<<<dim3(gsize), dim3(bsize)>>>(device_buffer->ptr, buffer->ptr, size * size);

            cudaMemcpy(this->host_buffer->ptr, this->device_buffer->ptr, sizeof(sf::Uint8) * size * size * 4, cudaMemcpyDeviceToHost);

            this->window->clear(sf::Color::Black);
            this->texture.update(this->host_buffer->ptr);
		    this->sprite.setTexture(this->texture);
		    this->sprite.setScale({2, 2});
		    this->window->draw(this->sprite);
		    this->window->display();
        }

        ~Window() {
            delete this->window;
            delete this->  host_buffer;
            delete this->device_buffer;
        }


};

