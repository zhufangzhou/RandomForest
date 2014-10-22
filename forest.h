#ifndef __FOREST_H
#define __FOREST_H

#include "tree.h"
#include "dataset.h"
#include <vector>
#include <string>

class BaseForest {
	private:
		void init(int n_trees, int n_threads, int max_depth, int min_leaf_samples);
	protected:
		// default values for this model
		static const int DEFAULT_N_THREADS = 1;
		static const int DEFAULT_N_TREES = 10;
		static const int DEFAULT_MAX_DEPTH = -1;
		static const int DEFAULT_MIN_LEAF_SAMPLES = 1;

		int n_classes;
		int n_threads;
		int n_trees;	
		int max_depth;
		int min_leaf_samples;
		int max_feature;

		Dataset *ds;
		std::vector<BaseTree> forest;
	public:
		BaseForest();
		BaseForest(int n_trees, int n_threads = DEFAULT_N_THREADS, int max_depth = DEFAULT_MAX_DEPTH, int min_leaf_samples = DEFAULT_MIN_LEAF_SAMPLES);
		
		~BaseForest();
};

class RandomForestClassifier : public BaseForest {
	private: 
		void init(int n_trees, int n_threads, int max_depth, int min_leaf_samples);
		int compute_max_feature(std::string max_feature_critreion, int feature_size);
		void build_forest(int max_feature, double* class_weight);
		void gen_model();
	public:
		RandomForestClassifier();
		RandomForestClassifier(int n_trees, int n_threads = DEFAULT_N_THREADS, int max_depth = DEFAULT_MAX_DEPTH, int min_leaf_samples = DEFAULT_MIN_LEAF_SAMPLES);
		void train(std::string filename, int feature_size, bool is_text, 
				int* discrete_idx = NULL, int discrete_size = 0, double* class_weight = NULL);
		void train(std::string feature_filename, std::string label_filename, int feature_size, int max_feature,
				int* discrete_idx = NULL, int discrete_size = 0, double* class_weight = NULL);
};
#endif
