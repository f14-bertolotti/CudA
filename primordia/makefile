
bin/primordia: $(shell find ../src/utils -type f -name "*.cpp") $(shell find ../src/utils -type f -name "*.cu") primordia.cpp makefile
	mkdir -p bin
	nvcc -x cu -O3 primordia.cpp -o bin/primordia -g -G -lsfml-graphics -lsfml-window -lsfml-system -lcufft

run: bin/primordia
	./bin/primordia

clean: 
	rm -rf bin
