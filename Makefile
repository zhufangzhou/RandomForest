main: main.o dataset.o Utils.o tree.o
	g++ Utils.o dataset.o tree.o main.o -o main -std=c++0x 
main.o: main.cpp
	g++ -c main.cpp -Wno-write-strings -std=c++0x
dataset.o: dataset.h dataset.cpp
	g++ -c dataset.cpp -std=c++0x
Utils.o: ../utils/Utils.cpp
	g++ -c ../utils/Utils.cpp -std=c++0x
tree.o: tree.h tree.cpp
	g++ -c tree.cpp -std=c++0x

clean:
	rm main.o dataset.o Utils.o tree.o main
