
# basic fft but with real-to-comple and complex-to-real
#bin/gol/golfftRC: src/gol/kernels.cu src/gol/golfftRC.cu makefile
#	nvcc -O3 src/gol/golfftRC.cu -o bin/gol/golfftRC -g -G -lcsfml-graphics -lcufft
#
#run-golfftRC: bin/gol/golfftRC
#	./bin/gol/golfftRC
#
## basic fft
#bin/gol/golfft: src/gol/kernels.cu src/gol/golfft.cu makefile
#	nvcc -O3 src/gol/golfft.cu -o bin/gol/golfft -g -G -lcsfml-graphics -lcufft
#
#run-golfft: bin/gol/golfft
#	./bin/gol/golfft
#
## basic convulution
#bin/gol/gol: src/gol/kernels.cu src/gol/main.cu makefile
#	nvcc -O3 src/gol/main.cu -o bin/gol/gol -g -G -lcsfml-graphics
#
#run-gol: bin/gol/gol
#	./bin/gol/gol

# larger than life fft
#bin/ltl/ltlfft: src/ltl/ltlfft.cu makefile
#	nvcc -O3 src/ltl/ltlfft.cu -o bin/ltl/ltlfft -g -G -lcsfml-graphics -lcufft
#
#run-ltlfft: bin/ltl/ltlfft
#	./bin/ltl/ltlfft
#
## basic fft with improved display
#bin/ltl/ltlfft-tex: src/ltl/ltlfft-tex.cu makefile
#	nvcc -O3 src/ltl/ltlfft-tex.cu -o bin/ltl/ltlfft-tex -g -G -lsfml-graphics -lsfml-window -lsfml-system -lcufft
#
#run-ltlfft-tex: bin/ltl/ltlfft-tex
#	./bin/ltl/ltlfft-tex
#
## primordia fft
#bin/primordia/primordia-fft: src/primordia/primordia-fft.cu makefile
#	nvcc -O3 src/primordia/primordia-fft.cu -o bin/primordia/primordia-fft -g -G  -lsfml-graphics -lsfml-window -lsfml-system -lcufft
#
#run-primordia-fft: bin/primordia/primordia-fft
#	./bin/primordia/primordia-fft
#
## states/time continuos primordia fft
#bin/primordia/continuos-primordia-fft: src/primordia/continuos-primordia-fft.cu makefile
#	nvcc -O3 src/primordia/continuos-primordia-fft.cu -o bin/primordia/continuos-primordia-fft -g -G  -lsfml-graphics -lsfml-window -lsfml-system -lcufft
#
#run-continuos-primordia-fft: bin/primordia/continuos-primordia-fft
#	./bin/primordia/continuos-primordia-fft
#
## states/time continuos primordia fft with variable kernel
#bin/primordia/continuos-primordia-R-fft: src/primordia/continuos-primordia-R-fft.cu makefile
#	nvcc -O3 src/primordia/continuos-primordia-R-fft.cu -o bin/primordia/continuos-primordia-R-fft -g -G  -lsfml-graphics -lsfml-window -lsfml-system -lcufft
#
#run-continuos-primordia-R-fft: bin/primordia/continuos-primordia-R-fft
#	./bin/primordia/continuos-primordia-R-fft
#
#
## states/time continuos primordia fft with round kernel
#bin/primordia/continuos-primordia-R-round-fft: src/primordia/continuos-primordia-R-round-fft.cu makefile
#	nvcc -O3 -x cu src/primordia/continuos-primordia-R-round-fft.cu -o bin/primordia/continuos-primordia-R-round-fft -g -G  -lsfml-graphics -lsfml-window -lsfml-system -lcufft
#
#run-continuos-primordia-R-round-fft: bin/primordia/continuos-primordia-R-round-fft
#	./bin/primordia/continuos-primordia-R-round-fft
#
##########################################################################################
##################################### THE C++ ERA ########################################
##########################################################################################

# GAME OF LIFE
bin/gol/golcpp: src/utils/* src/gol/gol.cpp makefile
	nvcc -x cu -O3 src/gol/gol.cpp -o bin/gol/golcpp -g -G -lsfml-graphics -lsfml-window -lsfml-system -lcufft

run-golcpp: bin/gol/golcpp
	./bin/gol/golcpp



# basic fft
#bin/gol/golfft: src/gol/kernels.cu src/gol/golfft.cu makefile
#	nvcc -O3 src/gol/golfft.cu -o bin/gol/golfft -g -G -lcsfml-graphics -lcufft
#
#run-golfft: bin/gol/golfft
#	./bin/gol/golfft
#
#
#
#bin/lenia/lenia: src/lenia/lenia.cpp makefile src/utils/*.cpp
#	nvcc -O3 -x cu src/lenia/lenia.cpp -o bin/lenia/lenia -g -G -lsfml-graphics -lsfml-window -lsfml-system -lcufft
#
#run-lenia: bin/lenia/lenia
#	./bin/lenia/lenia




