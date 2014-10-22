#include "forest.h"
#include "utils.h"
#include <cmath>

BaseForest::BaseForest() {
	BaseForest(DEFAULT_N_TREES, DEFAULT_N_THREADS, DEFAULT_MAX_DEPTH, DEFAULT_MIN_LEAF_SAMPLES);		
}

BaseForest::BaseForest(int n_trees, int n_threads, int max_depth, int min_leaf_samples) {
	init(n_trees, n_threads, max_depth, min_leaf_samples);
}

BaseForest::~BaseForest() {

}

void BaseForest::init(int n_trees, int n_threads, int max_depth, int min_leaf_samples) {
	this->n_trees = n_trees;
	this->n_threads = n_threads;
	this->max_depth = max_depth;
	this->min_leaf_samples = min_leaf_samples;
	this->n_classes = 2;
}

RandomForestClassifier::RandomForestClassifier() 
	: BaseForest(DEFAULT_N_TREES, DEFAULT_N_THREADS, DEFAULT_MAX_DEPTH, DEFAULT_MIN_LEAF_SAMPLES) {

}

RandomForestClassifier::RandomForestClassifier(int n_trees, int n_threads, int max_depth, int min_leaf_samples) 
	: BaseForest(n_trees, n_threads, max_depth, min_leaf_samples) {

}

int RandomForestClassifier::compute_max_feature(std::string max_feature_critreion, int feature_size) {
	double percent;
	if (max_feature_critreion == "sqrt") {
		return sqrt((double)feature_size);	
	} else if (max_feature_critreion == "log") {
		return log((double)feature_size);
	} else {
		percent = atof(max_feature_critreion.c_str());
		if (percent <= 1 || percent > 0) {
			return percent * feature_size;
		} else {
			throw "forest.cpp::compute_max_feature-->\n\t`max_feature_criterion` must be \"sqrt\" or \"log\" or double value between 0 and 1";
		}
	}
}

void RandomForestClassifier::train(std::string filename, int feature_size, bool is_text,
		int* discrete_idx, int discrete_size, double* class_weight) {
	// if not specify class_weight, use ones vector as default
	if (class_weight == NULL) 
		class_weight = gen_dones(n_classes);

	// read data from file
	if (is_text) {
		ds->readText(filename, feature_size, TRAIN, discrete_idx, discrete_size);
	} else {
		ds->readBinary(filename, feature_size, TRAIN, discrete_idx, discrete_size);
	}

	// train model
	build_forest(max_feature, class_weight);
	// generate model using tree structure
	gen_model();
}

void RandomForestClassifier::build_forest(int max_feature, double* class_weight) {

}

void RandomForestClassifier::gen_model() {

}

void RandomForestClassifier::train(std::string feature_filename, std::string label_filename, int feature_size, int max_feature,
		int* discrete_idx, int discrete_size, double* class_weight) {
	// if not specify class_weight, use ones vector as default
	if (class_weight == NULL)
		class_weight = gen_dones(n_classes);

	// read data from feature file and label file (only for binary file)
	ds->readBinary(feature_filename, label_filename, feature_size, discrete_idx, discrete_size);

	// train model
	build_forest(max_feature, class_weight);
	// generate model using tree strcture
	gen_model();
}

