#include <iostream>
#include <vector> 
#include <string>

#include "forest.h"
#include "dataset.h"
#include "tree.h"
#include "metrics.h"

dataset* mushroom() {
	float* weight = new float[2];
	weight[0] = weight[1] = 1.0;
	dataset* d = new dataset(2, 112, weight);
	d->load_data("./data/mushrooms", TRAIN);
	delete[] weight;
	return d;
}

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
	decision_tree* t = new decision_tree("6", 10, 1, 1);
	t->debug(d);

}

void test_decision_tree() {
	dataset *d = mushroom();
//	d->debug();
	decision_tree* t = new decision_tree("sqrt", 10000, 1, 1);
	//t->debug(d);
	// train the model
	t->build(d);
	t->export_dotfile("./display/tree.dot");
	data_reader* dr = new data_reader("./data/mushrooms", 112, TRAIN);
	std::vector<example_t*> test_data = dr->read_examples();	
	int n_test = test_data.size();
	float* y_pred_zero = t->predict_proba(test_data);	
	float* y_pred = y_pred_zero + n_test;
	int* y_true = new int[n_test];
	for (int i = 0; i < n_test; i++) y_true[i] = test_data[i]->y;
	std::string color_info = "yellow", color_value = "red";
	std::cout << color_msg("Precision = ", color_info ) << color_msg(Metrics::precision(y_pred, y_true, n_test), color_value) << std::endl;
	std::cout << color_msg("Recall = ", color_info) << color_msg(Metrics::recall(y_pred, y_true, n_test), color_value) << std::endl;
	std::cout << color_msg("F1-score = ", color_info) << color_msg(Metrics::f1_score(y_pred, y_true, n_test), color_value) << std::endl;
	std::cout << color_msg("AUC = ", color_info) << color_msg(Metrics::roc_auc_score(y_pred, y_true, n_test), color_value) << std::endl;
	std::cout << color_msg("Precision-Recall AUC = ", color_info) << color_msg(Metrics::pr_auc_score(y_pred, y_true, n_test), color_value) << std::endl;

	int n_features = d->get_n_features();
	float *imp = t->compute_importance();
	for (int i = 0; i < n_features; i++)
		std::cout << i+1 << ":" << imp[i] << " ";
	std::cout << std::endl;


	delete dr;
	delete t;
	delete d;
	delete y_true;
	delete y_pred_zero;
}

void test_random_forest() {
	dataset* d = mushroom();

	random_forest_classifier* rf = new random_forest_classifier("sqrt", 5, 1, 10, -1);
	rf->build(d);
	rf->export_dotfile("display/forest.dot", WHOLE_FOREST);
	rf->dump("model/forest.model");

	delete d;
	delete rf;
}

int main(int argc, char** argv) {
	//debug_data_reader();
	//debug_dataset();
	//debug_decision_tree();
	//test_decision_tree();
	test_random_forest();
	return 0;
}
