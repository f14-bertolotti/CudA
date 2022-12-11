
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

# larger than life fft
bin/ltl/ltlfft: src/ltl/ltlfft.cu makefile
	nvcc -O3 src/ltl/ltlfft.cu -o bin/ltl/ltlfft -g -G -lcsfml-graphics -lcufft

run-ltlfft: bin/ltl/ltlfft
	./bin/ltl/ltlfft

# basic fft with improved display
bin/ltl/ltlfft-tex: src/ltl/ltlfft-tex.cu makefile
	nvcc -O3 src/ltl/ltlfft-tex.cu -o bin/ltl/ltlfft-tex -g -G -lsfml-graphics -lsfml-window -lsfml-system -lcufft

run-ltlfft-tex: bin/ltl/ltlfft-tex
	./bin/ltl/ltlfft-tex

# primordia fft
bin/primordia/primordia-fft: src/primordia/primordia-fft.cu makefile
	nvcc -O3 src/primordia/primordia-fft.cu -o bin/primordia/primordia-fft -g -G  -lsfml-graphics -lsfml-window -lsfml-system -lcufft

run-primordia-fft: bin/primordia/primordia-fft
	./bin/primordia/primordia-fft



