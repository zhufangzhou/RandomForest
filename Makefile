CC = g++
ALL_LIB = main.o dataset.o tree.o utils.o
CXXFLAGS = -g -Wno-write-strings -std=c++0x -I../utils/include

main: $(ALL_LIB)
	$(CC) $(CXXFLAGS) $(ALL_LIB) -o main

%.o: %.cpp
	g++ -g -std=c++0x -c $< -o $@ -I../utils/include
utils.o: ../utils/src/utils.cpp
	g++ -g -std=c++0x -c ../utils/src/utils.cpp -o utils.o -I../utils/include
clean:
	rm -f main.o dataset.o  tree.o main utils.o
