
bin/lenia: $(shell find ../utils -type f -name "*.cpp") $(shell find ../utils -type f -name "*.cu") lenia.cpp makefile
	mkdir -p bin
	nvcc -x cu -O3 lenia.cpp -o bin/lenia -g -G -lsfml-graphics -lsfml-window -lsfml-system -lcufft

run: bin/lenia
	./bin/lenia

clean: 
	rm -rf bin
