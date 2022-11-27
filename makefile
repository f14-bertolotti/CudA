
bin/gol: src/main.cu
	nvcc src/main.cu -o bin/gol -g -G -lcsfml-graphics

run: bin/gol
	./bin/gol

