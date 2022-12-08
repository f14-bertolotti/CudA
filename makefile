
# basic fft but with real-to-comple and complex-to-real
bin/gol/golfftRC: src/gol/kernels.cu src/gol/golfftRC.cu makefile
	nvcc -O3 src/gol/golfftRC.cu -o bin/gol/golfftRC -g -G -lcsfml-graphics -lcufft

run-golfftRC: bin/gol/golfftRC
	./bin/gol/golfftRC

# basic fft
bin/gol/golfft: src/gol/kernels.cu src/gol/golfft.cu makefile
	nvcc -O3 src/gol/golfft.cu -o bin/gol/golfft -g -G -lcsfml-graphics -lcufft

run-golfft: bin/gol/golfft
	./bin/gol/golfft

# basic convulution
bin/gol/gol: src/gol/kernels.cu src/gol/main.cu makefile
	nvcc -O3 src/gol/main.cu -o bin/gol/gol -g -G -lcsfml-graphics

run-gol: bin/gol/gol
	./bin/gol/gol


gol-bins: bin/gol/golfftRC bin/gol/golfft bin/gol/gol
