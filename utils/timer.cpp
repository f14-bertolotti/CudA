#pragma once

#include <cstdlib>

class Timer {

    private:
        clock_t      start_value;
        clock_t      end_value;
        float        average;
        float        alpha;
        int          next;
        unsigned int iteration;


    public:
        Timer() {
            this->start_value = 0;
            this->end_value   = 0;
            this->alpha     = 0.01f;
            this->average   = 0.0f;
            this->iteration = 0;
        };

        void start() {
            start_value = clock();
        }

        void stop() {
            end_value = clock();
            next = ((end_value - start_value) * 1000 / CLOCKS_PER_SEC) % 1000;
            average = (alpha * next) + (1.0 - alpha) * average;
            ++iteration;
        }

        void print() {
            printf("\r==== itr: %10d, msc: %3d, avg: %.5f ====", iteration, next, average);
        }

};
