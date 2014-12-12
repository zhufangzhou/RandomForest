#include <iostream>
#include <vector> 
#include <string>

#include "dataset.h"
#include "tree.h"
void debug_data_reader() {
	data_reader* dr = new data_reader("data/train.dat", 10, TRAIN);
	example_t* single;
	std::vector<example_t*> ex_vec;

	//single = dr->read_an_example();
	ex_vec = dr->read_examples();
	for (auto it = ex_vec.begin(); it != ex_vec.end(); it++)
		(*it)->debug();
}

void debug_dataset() {
	float* weight = new float[2];
	weight[0] = weight[1] = 1.0;
	dataset* dd = new dataset(2, 6, weight);
	//dd->load_data("data/predict.dat", PREDICT);	
	dd->load_data("data/train.dat", TRAIN);	
	dd->debug();

	delete[] weight;
	delete dd;
}

void debug_decision_tree() {
	float* weight = new float[2];
	weight[0] = weight[1] = 1.0;
	dataset* d = new dataset(2, 6, weight);

	d->load_data("./data/train.dat", TRAIN);
	d->debug();
	decision_tree* t = new decision_tree("6", 10, 1);
	t->debug(d);

}

void test_decision_tree() {
	float* weight = new float[2];
	weight[0] = weight[1] = 1.0;
	dataset* d = new dataset(2, 112, weight);
	d->load_data("./data/mushrooms", TRAIN);
	d->debug();
	decision_tree* t = new decision_tree("100", 10000, 1);
	t->debug(d);

	delete[] weight;
	delete t;
	delete d;
}

int main(int argc, char** argv) {
	//debug_data_reader();
	//debug_dataset();
	//debug_decision_tree();
	test_decision_tree();
	return 0;
}
