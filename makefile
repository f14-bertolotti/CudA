
bin/golfft: src/kernels.cu src/golfft.cu makefile
	nvcc -O3 src/golfft.cu -o bin/golfft -g -G -lcsfml-graphics -lcufft

run-golfft: bin/golfft
	./bin/golfft

bin/gol: src/kernels.cu src/main.cu makefile
	nvcc -O3 src/main.cu -o bin/gol -g -G -lcsfml-graphics

run-gol: bin/gol
	./bin/gol

