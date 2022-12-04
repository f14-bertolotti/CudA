
bin/gol: src/kernels.cu src/main.cu makefile
	nvcc src/main.cu -o bin/gol -g -G -lcsfml-graphics

run-gol: bin/gol
	./bin/gol

