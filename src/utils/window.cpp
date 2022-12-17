#pragma once

#include <SFML/Graphics.hpp>

class Window {

    public:
        sf::Texture     texture;
        sf::Sprite       sprite;

        sf::RenderWindow*       window;
        std::vector<sf::Uint8>* buffer;

        Window(int size) {
            this->window = new sf::RenderWindow(sf::VideoMode(size, size), "cellular automaton in cuda");
	        this->buffer = new std::vector<sf::Uint8>(size * size * 4);
	        this->texture.create(size, size);
        }

        sf::Uint8* get_buffer() {
            return buffer->data();
        }

        void update() {
		    this->sprite.setTexture(this->texture);
		    this->sprite.setScale({2, 2});
		    this->window->draw(this->sprite);
		    this->window->display();
        }

        ~Window() {
            delete this->buffer;
            delete this->window;
        }


};

//    sf::RenderWindow window(sf::VideoMode(GSIZE, GSIZE), "larger than life fft");
//    sf::Texture texture;
//	sf::Sprite sprite;
//	std::vector<sf::Uint8> pixelBuffer(GSIZE * GSIZE * 4);
//	texture.create(GSIZE, GSIZE);

//        texture.update(pixelBuffer.data());
//		sprite.setTexture(texture);
//		sprite.setScale({2, 2});
//		window.draw(sprite);
//		window.display();


