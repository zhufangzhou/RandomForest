/**
 * @file forest.h
 * @brief header file of forest, including random forest
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-12-21
 */
#pragma once

/* C header file */
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
/* C++ header file */
#include <iostream>
#include <vector>
#include <string>
#include <thread>
/* my header file */
#include "parameter.h"
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
		std::string feature_rule;

		int max_feature;
		int max_depth;
		int min_split;

		float* fea_imp;

		bool is_build;
		
		bool check_build();

		void free_forest();
		void parallel_predict_proba(int tree_begin, int tree_end, std::vector<example_t*> &examples, float* ret);
		void parallel_apply(int tree_begin, int tree_end, std::vector<example_t*> &examples, int* ret);
	public:
		forest();
		forest(const std::string feature_rule, int max_depth, int min_split, int n_trees, int n_threads);
		virtual ~forest();
		float* compute_importance(bool re_compute = false);
		int* apply(std::vector<example_t*> &examples);
		float* predict_proba(std::vector<example_t*> &examples);
		int* predict_label(std::vector<example_t*> &examples);
		int* get_leaf_counts();
		int get_max_feature();
		int get_n_features();
		virtual void dump(const std::string& filename) const = 0;
		virtual void load(const std::string& filename) = 0;
};

class random_forest_classifier : forest {
	private:
		void parallel_build(int tree_begin, int tree_end, dataset*& d);
	public:
		random_forest_classifier(const std::string feature_rule, int max_depth, int min_split, int n_trees, int n_threads);
		random_forest_classifier();
		~random_forest_classifier();

		void build(dataset*& d);
		void print_info();
		void dump(const std::string& filename) const;
		void load(const std::string& filename);
		void debug(dataset*& d);
};
