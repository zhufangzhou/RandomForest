#ifndef TREE_HEADER
#define TREE_HEADER

#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include "dataset.h"

typedef int sample_size_t;

// tree node structure
struct node_t {
	double weighted_frequency[2];			// 0 is negative, 1 is positive
	// sample index in this node, will be deleted after splitting
	sample_size_t *sample_index;
	// sample size in this node
	sample_size_t sample_size;
	
	// the chosen feature to split the node
	int feature_index;
	// whether this feature is discrete
	bool is_discrete;
	// the threshold of the chosen feature
	double feature_value;

	// depth of this node in the tree
	int depth;
	// whether this node is leaf
	bool is_leaf;

	// index of left child in the tree vector
	int lchild;
	// index of right child in the tree vector
	int rchild;

	void reset() {
		sample_index = NULL;
		memset(weighted_frequency, 0, sizeof(double)*2);
		feature_index = -1;
		feature_value = 0;
		is_leaf = false;
		lchild = rchild = -1;
		depth = -1;
	}

	node_t() {
		reset();
	}
};

class BaseTree {
protected:
	virtual void build_tree() = 0;
	Dataset *ds;
	int max_depth;								// max depth of the tree
	int min_leaf_samples;						// minimum samples in the leaves
	int n_classes;
	std::vector<node_t> tree;						// tree structure
	void check_param(int min_leaf_sample, int max_depth, int feature_size);
	void init(int min_leaf_samples, int max_depth);
public:
	// read data from binary file
	BaseTree(std::string filename, int feature_size, int min_leaf_samples, int max_depth);
	BaseTree(std::string feature_filename, std::string label_filename, int feature_size, 
		int min_leaf_samples, int max_depth);
	// read data from text file
	BaseTree(std::string filename, int min_leaf_samples, int max_depth);
	~BaseTree();
	void train(double *class_weight = NULL);
	void train(double *X, int *y, double *class_weight = NULL);
	//double* predict(std::string filename);	// binary
	//double* predict(std::string filename);	// text
};

class DecisionTreeClassifier : public BaseTree {
protected:
	void build_tree(int max_feature, double *class_weight);
	void split(int max_feature, int *feature_list, double *class_weight, node_t *pa, node_t *lchilde, node_t *rchild);
public:
	DecisionTreeClassifier(std::string filename, int feature_size, int min_leaf_samples, int max_depth);
	DecisionTreeClassifier(std::string filename, int min_leaf_samples, int max_depth);
};

class Criterion {
public:
	static double gini(double *arr, int size);
};

#endif