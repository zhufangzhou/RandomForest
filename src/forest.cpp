#include "forest.h"
#include "utils.h"
#include "parallel.h"
#include <cmath>

const std::string BaseForest::DEFAULT_MAX_FEATURE_CRITERION = "sqrt";

BaseForest::BaseForest() {
	BaseForest(DEFAULT_N_TREES, DEFAULT_N_THREADS, DEFAULT_MAX_FEATURE_CRITERION, DEFAULT_MAX_DEPTH, DEFAULT_MIN_LEAF_SAMPLES);		
}

BaseForest::BaseForest(int n_trees, int n_threads, std::string max_feature_criterion, int max_depth, int min_leaf_samples) {
	init(n_trees, n_threads, max_feature_criterion, max_depth, min_leaf_samples);
}

BaseForest::~BaseForest() {

}

void BaseForest::check_param(int n_trees, int n_threads, std::string max_feature_criterion, int max_depth, int min_leaf_samples) {
	// check `n_tree`
	if (n_trees < 1) 
		throw "forest.cpp::check_param-->\n\t`n_trees` must satisfy `n_trees` >= 1";
	// check `n_threads`
	if (n_threads < 1)
		throw "forest.cpp::check_param-->\n\t`n_threads` must satisfy `n_threads` >= 1";
	// check `max_feature_criterion`
	if (max_feature_criterion != "sqrt" && max_feature_criterion != "log" && is_number(max_feature_criterion)) {
		throw "forest.cpp::check_param-->\n\t`max_feature_criterion` must be 'sqrt' or 'log' or real number which is between 0 and 1 or positive integer value which is less than `feature size`";
	}
	// check `max_depth`
	if (max_depth != -1 && max_depth < 2) {
		throw "forest.cpp::check_param-->\n\t`max_depth` must satisfy `max_depth` >= 2 or `max_depth` = -1 (means do not control tree's growth)";	
	}
	// check `min_leaf_samples`
	if (min_leaf_samples < 1) {
		throw "forest.cpp::check_param-->\n\t`min_leaf_samples` must satisfy `min_leaf_samples` >= 1";
	}
}

void BaseForest::init(int n_trees, int n_threads, std::string max_feature_criterion, int max_depth, int min_leaf_samples) {
	this->n_trees = n_trees;
	this->n_threads = n_threads;
	this->max_feature_criterion = max_feature_criterion;
	this->max_depth = max_depth;
	this->min_leaf_samples = min_leaf_samples;
	this->n_classes = 2;
}

RandomForestClassifier::RandomForestClassifier() 
	: BaseForest(DEFAULT_N_TREES, DEFAULT_N_THREADS, DEFAULT_MAX_FEATURE_CRITERION, DEFAULT_MAX_DEPTH, DEFAULT_MIN_LEAF_SAMPLES) {

}

RandomForestClassifier::RandomForestClassifier(int n_trees, int n_threads, std::string max_feature_criterion, int max_depth, int min_leaf_samples) 
	: BaseForest(n_trees, n_threads, max_feature_criterion, max_depth, min_leaf_samples) {

}

int RandomForestClassifier::compute_max_feature(int feature_size) {
	double percent;
	if (max_feature_criterion == "sqrt") {
		return sqrt((double)feature_size);	
	} else if (max_feature_criterion == "log") {
		return log((double)feature_size);
	} else {
		percent = atof(max_feature_criterion.c_str());
		if (percent <= 1 || percent > 0) {
			return (int)(percent * feature_size);
		} else if (percent == (int)percent && percent > 1 && percent <= feature_size) {
			return (int)percent;
		} else {
			throw "forest.cpp::compute_max_feature-->\n\t`max_feature_criterion` must be \"sqrt\" or \"log\" or double value between 0 and 1 or positive integer value which are less than `feature size`";
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

	// compute max_feature
	this->max_feature = compute_max_feature (feature_size);

	// train model
	build_forest(class_weight);
}

void RandomForestClassifier::train(std::string feature_filename, std::string label_filename, int feature_size, int max_feature,
		int* discrete_idx, int discrete_size, double* class_weight) {
	// if not specify class_weight, use ones vector as default
	if (class_weight == NULL)
		class_weight = gen_dones(n_classes);

	// read data from feature file and label file (only for binary file)
	ds->readBinary(feature_filename, label_filename, feature_size, discrete_idx, discrete_size);

	// compute max_feature
	this->max_feature = compute_max_feature (feature_size);

	// train model
	build_forest(class_weight);
}

void RandomForestClassifier::parallel_build_forest(int tree_start, int tree_end, double* class_weight) {
	for (int i = tree_start; i < tree_end; i++) {
		forest[i] = new DecisionTreeClassifier(min_leaf_samples, max_depth);			
		forest[i]->train(ds, max_feature, class_weight);
	}
}

void RandomForestClassifier::build_forest(double* class_weight) {
	int tree_start = 0, tree_end;
	
	// reset the forest structure
	forest.clear();
	// configure the forest structure size
	forest.resize(n_trees);

	// assign works to each thread
	struct parallel_unit pu = init_block(n_trees, n_threads);		
	std::vector<std::thread> threads(pu.num_threads - 1);
	for (int i = 0; i < pu.num_threads; i++) {
		tree_end = tree_start + pu.block_size;	
		// create a new thread to build some trees
		threads[i] = std::thread([this, tree_start, tree_end, class_weight]() {
				parallel_build_forest(tree_start, tree_end, class_weight);
		});
		tree_start = tree_end;
	}
	// call the last block in the parent thread
	parallel_build_forest(tree_start, n_trees, class_weight);
	// waiting for other threads to finish
	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
}


