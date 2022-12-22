
bin/gol: $(shell find ../src/utils -type f -name "*.cpp") $(shell find ../src/utils -type f -name "*.cu") gol.cpp makefile
	mkdir -p bin
	nvcc -x cu -O3 gol.cpp -o bin/gol -g -G -lsfml-graphics -lsfml-window -lsfml-system -lcufft

run: bin/gol
	./bin/gol

clean: 
	rm -rf bin
