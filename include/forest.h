/**
 * @file forest.h
 * @brief 
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-12-21
 */
#pragma once

// C++ header file
#include <vector>
#include <string>
#include <thread>
// my header file
#include "tree.h"
#include "dataset.h"
#include "parallel.h"

/* declaration */
class forest;
class random_forest_classifier;

class forest {
	protected:
		std::vector<tree*> trees;

		int n_trees;
		int n_threads;
		int n_classes;
		int n_features;
		const std::string feature_rule;

		int max_feature;
		int max_depth;
		int min_split;

		float* fea_imp;
		
		void check_build();

		void parallel_predict_proba(int tree_begin, int tree_end, std::vector<example_t*> &examples, float* ret);
		void parallel_apply(int tree_begin, int tree_end, std::vector<example_t*> &examples, int* ret);
	public:
		forest();
		~forest();
		void init();
		float* compute_importance(bool re_compute = false);
		int* apply(std::vector<example_t*> &examples);
		float* predict_proba(std::vector<example_t*> &examples);
		int* predict_label(std::vector<example_t*> &examples);
		void free_forest();
		int* get_leaf_counts();
		int get_max_feature();
		int get_n_features();
};

class random_forest_classifier : forest {
	protected:

	public:
		random_forest_classifier(const std::string feature_rule, int max_depth, int min_split);

		void build(dataset*& d);
		void print_info();
		void dump(const std::string& filename);
		void load(const std::string& filename);
		void debug(dataset*& d);
};
