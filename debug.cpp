#include <iostream>
#include <vector> 
#include <string>

#include "dataset.h"

void debug_DataReader() {
	DataReader* dr = new DataReader("data/train.dat", 10, TRAIN);
	example_t* single;
	std::vector<example_t*> ex_vec;

	//single = dr->read_an_example();
	ex_vec = dr->read_examples();
	for (auto it = ex_vec.begin(); it != ex_vec.end(); it++)
		(*it)->debug();
}

void debug_Dataset() {
	Dataset* dd = new Dataset(2, 6);
	//dd->load_data("data/predict.dat", PREDICT);	
	dd->load_data("data/train.dat", TRAIN);	
	dd->debug();
}

int main(int argc, char** argv) {
	//debug_DataReader();
	debug_Dataset();
	return 0;
}
