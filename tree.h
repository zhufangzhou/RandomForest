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
#include "utils.h"

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


/**
 * @brief An abstract class for node in the tree
 */
class node {
	public:
		bool is_cate; /** is the split feature categorical */
		int feature_id; /** split feature id */
		feature_t threshold; /** for categorical attribute is the chosen feature value for left child node, for continous attribute is the threshold to determine left or right */
		float gain; /** heuristic measure(e.g. gini index or information gain) */

		float* cur_frequency; /** size should be `n_classes`, means the weighted frequency for each class */
		int n_classes; /** number of different class in the node */

		int leaf_idx; 	/** -1 if this node is not leaf otherwise non-negtive integer */
		node* left; /** point to left node */
		node* right; /** point to right node */

		/**
		 * @brief Constructor
		 *
		 * @param n_classes number of different class
		 */
		node(int n_classes);
		/**
		 * @brief Destructor
		 */
		~node();
		/**
		 * @brief Dump an single node to a binary file
		 *
		 * @param filename path to dump
		 */
		void dump(const std::string& filename);
		/**
		 * @brief Dump an single node to the output file stream
		 *
		 * @param ofs output file stream (should be open first, did not close in this function)
		 */
		void dump(std::ofstream& ofs);
};

/**
 * @brief Specify for batch tree algorithm (e.g. decision tree)
 */
class batch_node : public node {
	public:
		batch_node(int n_classes);

};

/**
 * @brief Specify for online tree algorithm 
 */
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

		/**
		 * @brief Non-parameter constructor (need to call `init` function maually if using this constructor
		 */
		tree();
		/**
		 * @brief Destructor
		 */
		~tree();
		/**
		 * @brief Constructor giving tree settings
		 *
		 * @param feature_rule number of feature to consider per node, avaiable values are 'sqrt' for square root of `n_features`, 'log' for logarithm of `n_features`, real number between 0 and 1 for percent of `n_features`, integer larger than 1 for fixed number features which should less than `n_features`. If the value is invalid, the program will take `sqrt` as default other than just exit.
		 * @param max_depth the depth limitation of tree 
		 * @param min_split the minimum number of examples needed to make a split in a node
		 */
		tree(const std::string feature_rule, int max_depth, int min_split);
		/**
		 * @brief Initialize the tree(e.g. set some parameter and allocate memory to some variables)
		 *
		 * @param feature_rule number of feature to consider per node, avaiable values are 'sqrt' for square root of `n_features`, 'log' for logarithm of `n_features`, real number between 0 and 1 for percent of `n_features`, integer larger than 1 for fixed number features which should less than `n_features`. If the value is invalid, the program will take `sqrt` as default other than just exit.
		 * @param max_depth the depth limitation of tree 
		 * @param min_split the minimum number of examples needed to make a split in a node
		 */
		void init(const std::string feature_rule, int max_depth, int min_split);
		/**
		 * @brief Compute feature importance after building the tree (should call build first)
		 *
		 * @param re_compute if `re_compute` set to true, then the importance will compulsively be re-computed. Otherwise, it will return the result computed before
		 *
		 * @return an float vector (size is `n_features`), each entry represent the corresponding feature's importance when building the tree (ps. all entry sum to one)
		 */
		float* compute_importance(bool re_compute = false);

		/**
		 * @brief Free memory space of the tree which use `root` as root node (only the tree structure)
		 *
		 * @param root root node of the tree to be free 
		 */
		void free_tree(node*& root);
		/**
		 * @brief Dump an single tree to an binary file
		 *
		 * @param filename path to dumped
		 */
		void dump(const std::string& filename);		
		/**
		 * @brief Load the tree from file
		 *
		 * @param filename path to load 
		 */
		void load(const std::string& filename);
		/**
		 * @brief Export the tree structure to a dot file, which can be used to generate a picture (dot -Tpng -o tree.png tree.dot)
		 *
		 * @param filename path to the dot file 
		 */
		void export_dotfile(const std::string& filename);

		/**
		 * @brief Return private member `max_feature` value
		 *
		 * @return max_feature computed according to `feature_rule`
		 */
		int get_max_feature();
};

/**
 * @brief A Decision Tree Classifier which is for sparse dataset
 */
class decision_tree : public tree {
	private:
		void build_rec(node*& root, dataset*& d, int depth);
		int verbose;
	public:
		decision_tree(const std::string feature_rule, int max_depth, int min_split);
		void build(dataset*& d);
		void debug(dataset*& d);
};

class online_tree : public tree {

};

class splitter {
	protected:
		virtual void update(int fea_id, float threshold, float*& left, node*& nd, criterion*& cr) = 0;
	public:
		int fea_id;					/** split feature id */
		float threshold;			/** split threshold */

		float gain;					/** heuristc measure (e.g. information gain or gini index) improvement after split */
		
		int n_classes; 				/** different classes when split */
		float* left_frequency; 		/** left[j] refers to weighted frequency for class j */
		float* right_frequency; 	/** right[j] refers to weighted frequency for class j */

		splitter(int n_classes);
		~splitter();
		virtual void split(tree* t, node*& root, dataset*& d, criterion*& cr) = 0;
};

class best_splitter : public splitter {
	protected:
		void update(int fea_id, float threshold, float*& left, node*& nd, criterion*& cr);
	public: 
		best_splitter(int n_classes);
		void split(tree* t, node*& root, dataset*& d, criterion*& cr);	
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
		gini(float*& frequency, int n_classes);
		float measure(float*& frequency, int n_classes);
};
