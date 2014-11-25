CC := g++
ALL_OBJ := $(patsubst %.cpp,%.o, $(wildcard *.cpp)) utils.o
CXXFLAGS := -g -Wno-write-strings -std=c++0x -I../utils/include

main: $(ALL_OBJ)
	$(CC) $(CXXFLAGS) $(ALL_OBJ) -o main

debug: debug.o dataset.o utils.o tree.o

utils.o: ../utils/src/utils.cpp
	g++ -g -std=c++0x -c ../utils/src/utils.cpp -o utils.o -I../utils/include

%.o: %.cpp
	g++ -g -std=c++0x -c $< -o $@ -I../utils/include
clean:
	rm -f main $(ALL_OBJ)
