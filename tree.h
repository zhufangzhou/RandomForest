/**
 * @file tree.h
 * @brief 
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-11-19
 */
#pragma once

/* C header file */
#include <cstdio>
#include <cstdlib>
#include <cmath>
/* C++ header file */
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <stack>
#include <iomanip>
/* my header file */
#include "dataset.h"

/* declaration */
class node;
class batch_node;
class online_node;
class tree;
class decision_tree;
class online_tree;
class splitter;
class best_splitter;
class random_splitter;
class criterion;
class gini;


class node {
	public:
		bool is_cate; /** is the split feature categorical */
		int feature_id; /** split feature id */
		feature_t threshold; /** for categorical attribute is the chosen feature value for left child node, for continous attribute is the threshold to determine left or right */
		float measure; /** heuristic measure(e.g. gini index or information gain) */
		int n_examples; /** number of examples in this node */

		float* cur_frequency; /** size should be `n_classes`, means the weighted frequency for each class */
		int n_classes; /** number of different class in the node */

		int leaf_idx; 	/** -1 if this node is not leaf otherwise non-negtive integer */
		node* left; /** point to left node */
		node* right; /** point to right node */

		node(int n_classes);
		~node();
		void dump(const std::string& filename);
		void dump(std::ofstream& ofs);
};

class batch_node : public node {
	public:
		batch_node(int n_classes);

};

class online_node : public node {
	public:
};

class tree {
	protected:
		node* root; 		/** root node of the tree */
		node** leaf_pt; 	/** pointer array which point to all the leaf in the tree */
		int leaf_size; 		/** number of leaves in the tree */
		
		int n_features; 	/** total number of features in the training set */
		std::string feature_rule; 	/** max feature criterion for splitting,
				 					* default `sqrt`, avaiable option are `log` or real number between 0 and 1
								    * represent percent of `n_features` or integer larger than 1 represent number of `max_feature`
									* */

		int max_feature; 	/** number of feature to consider when split */
		int max_depth; 		/** the maximum depth to grow */
		int min_split; 		/** the minimum examples needed to split */

		float* fea_imp; 	/** feature importance */
	public:
		int* valid; 		/** is the example valid to consider when split */

		tree();
		~tree();
		tree(const std::string feature_rule, int max_depth, int min_split);
		void init(const std::string feature_rule, int max_depth, int min_split);
		float* compute_importance(bool re_compute = false);

		void free_tree(node*& nd);
		void dump(const std::string& filename);		
		void load(const std::string& filename);
		void export_dotfile(const std::string& filename);

		int get_max_feature();
		int* get_valid();
};

class decision_tree : public tree {
	public:
		void build(dataset*& d);
};

class online_tree : public tree {

};

class splitter {
	public:
		int fea_id;					/** split feature id */
		float threshold;			/** split threshold */

		int gain;					/** heuristc measure (e.g. information gain or gini index) improvement after split */
		
		int n_classes; 				/** different classes when split */
		float* left_frequency; 		/** left[j] refers to weighted frequency for class j */
		float* right_frequency; 	/** right[j] refers to weighted frequency for class j */

		splitter(int n_classes);
		~splitter();
		virtual void split(tree*& t, node*& root, dataset*& d, criterion*& cr) = 0;
		virtual void update(int fea_id, float threshold, float*& left, node*& nd, criterion*& cr) = 0;
};

class best_splitter : public splitter {

	public: 
		void split(tree*& t, node*& root, dataset*& d, criterion*& cr);	
		void update(int fea_id, float threshold, float*& left, node*& nd, criterion*& cr);
};

class random_splitter : public splitter {

};

class criterion {
	protected:
		float tot_frequency; 	/** temporary variable store `tot_frequency` after calling `measure` function */
		float cur_measure; 		/** current node heuristic measure value */
		float cur_tot; 			/** current node total frequency */

		bool is_init; 			/** has set the current node measure */
	public:
		criterion();
		criterion(float*& frequency, int n_classes);

		void set_current(float*& frequency, int n_classes);
		float gain(float*& left_frequency, float*& right_frequency, int n_classes);

		virtual float measure(float*& frequency, int n_classes) = 0;
};

class gini : public criterion {
	public:
		float measure(float*& frequency, int n_classes);
};
