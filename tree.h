/**
 * @file tree.h
 * @brief 
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-11-19
 */
#pragma once

#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dataset.h"

class node {
	protected:
		bool is_cate; /** is the split feature categorical */
		int feature_id;
		feature_t threshold; /** for categorical attribute is the chosen feature value for left child node, for continous attribute is the threshold to determine left or right */

		float* portion; /** size should be n_classes, means the weighted frequency for each class */

		int leaf_idx; 	/** -1 if this node is not leaf otherwise non-negtive integer */
		node_t* left; /** point to left node */
		node_t* right; /** point to right node */
	public:
		void dump(const std::string& filename, int n_classes);
};

class batch_node : public node_t {
	public:

};

class online_node : public node_t {
	public:
};

class tree {
	protected:
		node_t* root; 		/** root node of the tree */
		node_t** leaf_pt; 	/** pointer array which point to all the leaf in the tree */
		int leaf_size; 		/** number of leaves in the tree */
		
		int max_feature; 	/** number of feature to consider when split */
		int max_depth; 		/** the maximum depth to grow */
		int min_split; 		/** the minimum examples needed to split */

		float* fea_imp; 	/** feature importance */
	public:
		tree_t();
		tree_t(std::string feature_rule, int max_depth, int min_split);
		void init(std::string feature_rule, int max_depth, int min_split);
		virtual void build(Dataset* d) = 0;
		float* compute_importance();

		void dump(std::string& filename);		
		void load(std::string& filename);
};

class decision_tree : public tree_t {

}

class online_tree : public tree_t {

};

class split_t {
	int fea_id; 	/** feature id */
};

class best_split : public split_t{

};

class random_split : public split_t {

};
