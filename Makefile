CC = g++
CXXFLAGS = -Wno-write-strings -std=c++0x -I../utils

main: main.o dataset.o Utils.o tree.o
main.o: main.cpp
dataset.o: dataset.h dataset.cpp
Utils.o: ../utils/Utils.cpp
	g++ -c ../utils/Utils.cpp -std=c++0x
tree.o: tree.h tree.cpp

clean:
	rm -f main.o dataset.o Utils.o tree.o main
