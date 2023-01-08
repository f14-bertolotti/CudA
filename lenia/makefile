
bin/lenia: $(shell find ../src/utils -type f -name "*.cpp") $(shell find ../src/utils -type f -name "*.cu") lenia.cpp makefile
	mkdir -p bin
	nvcc -x cu -O3 lenia.cpp -o bin/lenia -g -G -lsfml-graphics -lsfml-window -lsfml-system -lcufft

run: bin/lenia
	./bin/lenia

liblenia.o: $(shell find ../src/utils -type f -name "*.cpp") $(shell find ../src/utils -type f -name "*.cu") liblenia.cpp makefile
	nvcc -O3 -c -x cu -Xcompiler -fPIC -o liblenia.o liblenia.cpp -lcufft 

liblenia.so: liblenia.o
	gcc -shared -fPIC -o liblenia.so liblenia.o -lstdc++

clean: 
	rm -rf bin
	rm liblenia.o
	rm liblenia.so
