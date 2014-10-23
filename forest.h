#ifndef __FOREST_H
#define __FOREST_H

#include "tree.h"
#include "dataset.h"
#include <vector>
#include <string>

class BaseForest {
	private:
		void init(int n_trees, int n_threads, std::string max_feature_criterion, int max_depth, int min_leaf_samples);
		void check_param(int n_tree, int n_threads, std::string max_feature, int max_depth, int min_leaf_samples);
	protected:
		// default values for this model
		static const int DEFAULT_N_THREADS = 1;
		static const int DEFAULT_N_TREES = 10;
		static const int DEFAULT_MAX_DEPTH = -1;
		static const int DEFAULT_MIN_LEAF_SAMPLES = 1;
		static const std::string DEFAULT_MAX_FEATURE_CRITERION;

		/* these settings are forest settings */
		int n_threads; 								// number of threads to use when build forest
		int n_trees;								// the forest size
		std::string max_feature_criterion; 			// rule to comput max feature number
		int max_feature; 							// the max features to choose when conduct a split	

		/* these settings are tree settings */
		int n_classes; 								// number of different classes
		int max_depth; 								// maximum depth for each tree estimator
		int min_leaf_samples; 						// minimum samples in each leaf

		Dataset *ds; 								
		std::vector<BaseTree*> forest;
	public:
		BaseForest();
		BaseForest(int n_trees, int n_threads = DEFAULT_N_THREADS, std::string max_feature_criterion = "sqrt", int max_depth = DEFAULT_MAX_DEPTH, int min_leaf_samples = DEFAULT_MIN_LEAF_SAMPLES);
		
		~BaseForest();
};

class RandomForestClassifier : public BaseForest {
	private: 
		void init(int n_trees, int n_threads, int max_depth, int min_leaf_samples);
		int compute_max_feature(int feature_size);
		void parallel_build_forest(int tree_start, int tree_end, double* class_weight);
		void build_forest(double* class_weight);
	public:
		RandomForestClassifier();
		RandomForestClassifier(int n_trees, int n_threads = DEFAULT_N_THREADS, std::string max_feature_criterion = "sqrt", int max_depth = DEFAULT_MAX_DEPTH, int min_leaf_samples = DEFAULT_MIN_LEAF_SAMPLES);
		void train(std::string filename, int feature_size, bool is_text, 
				int* discrete_idx = NULL, int discrete_size = 0, double* class_weight = NULL);
		void train(std::string feature_filename, std::string label_filename, int feature_size, int max_feature,
				int* discrete_idx = NULL, int discrete_size = 0, double* class_weight = NULL);
};
#endif
