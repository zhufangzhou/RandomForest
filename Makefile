main: main.o dataset.o Utils.o tree.o
	g++ Utils.o dataset.o tree.o main.o -o main -std=c++0x 
main.o: main.cpp
	g++ -c main.cpp -Wno-write-strings
dataset.o: dataset.h dataset.cpp
	g++ -c dataset.cpp
Utils.o: ../utils/Utils.cpp
	g++ -c ../utils/Utils.cpp
tree.o: tree.h tree.cpp
	g++ -c tree.cpp

clean:
	rm main.o dataset.o Utils.o tree.o main
