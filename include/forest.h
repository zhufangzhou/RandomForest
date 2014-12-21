/**
 * @file forest.h
 * @brief 
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-12-21
 */
#pragma once

#include "tree.h"
#include "dataset.h"
#include <vector>
#include <string>
#include <thread>

/* declaration */
class forest;
class random_forest_classifier;

class forest {
	protected:
		vector<tree*> trees;

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
	public:
		forest();
		~forest();
		void init();
		float* compute_importance(bool re_compute = false);
		int* apply(std::vector<example_t*> &examples);
		float* predict_proba(std::vector<example_t*> &examples);
		int* predict_label(std::vector<example_t*> &examples);
		void free_forest();
		int get_max_feature();
		int get_n_features();
};

class random_forest_classifier : forest {
	protected:

	public:
		random_forest_classifier(const std::string feature_rule, int max_depth, int min_split);

		void build(dataset*& d);
		print_info();
		void dump(const std::string& filename);
		void load(const std::string& filename);
		void debug(dataset*& d);
};
