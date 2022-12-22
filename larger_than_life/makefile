
bin/ltl: $(shell find ../src/utils -type f -name "*.cpp") $(shell find ../src/utils -type f -name "*.cu") ltl.cpp makefile
	mkdir -p bin
	nvcc -x cu -O3 ltl.cpp -o bin/ltl -g -G -lsfml-graphics -lsfml-window -lsfml-system -lcufft

run: bin/ltl
	./bin/ltl

clean: 
	rm -rf bin
