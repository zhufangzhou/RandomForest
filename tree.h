#ifndef TREE_HEADER
#define TREE_HEADER

#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BINARY false
#define TEXT true
#define PROBA true
#define LABEL false
typedef int sample_size_t;

class Dataset;

// tree node structure
struct node_t {
	// whether this node is leaf
	bool is_leaf;

	double weighted_frequency[2];			// 0 is negative, 1 is positive
	// sample size in this node
	sample_size_t sample_size;
	// sample index in this node, will be deleted after splitting
	sample_size_t *sample_index;
	
	

	// the chosen feature to split the node
	int feature_index;
	// whether this feature is discrete
	bool is_discrete;
	// the threshold of the chosen feature
	double feature_value;

	double improvement;
	double criterion;

	// depth of this node in the tree
	int depth;
	

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
		criterion = -1;
	}

	node_t() {
		reset();
	}
};

// just part of node_t
struct model_t {
	bool is_leaf;
	double node_value;

	int feature_index;
	bool is_discrete;
	double feature_value;

	double improvement;
	double criterion;

	int depth;
	int lchild;
	int rchild;
	model_t() {

	}
	model_t(bool is_leaf, double node_value, int feature_index, bool is_discrete, double feature_value, 
		double improvement, double criterion,
		int depth, int lchild, int rchild) : is_leaf(is_leaf), node_value(node_value), feature_index(feature_index),
		 is_discrete(is_discrete), feature_value(feature_value), improvement(improvement), criterion(criterion), depth(depth), lchild(lchild), rchild(rchild) {}
};

struct model_meta {
	int model_size;
	int feature_size;
	model_t *tree;
};

class BaseTree {
protected:
	virtual void build_tree(int max_feature, double *class_weight) = 0;
	virtual void dump(std::string model_filename) = 0;
	virtual void load(std::string model_filename) = 0;
	virtual void gen_model() = 0;
	virtual void export_dotfile(std::string dot_filename) = 0;
	virtual double* compute_importance() = 0;
	Dataset *ds;
	int max_depth;								// max depth of the tree
	int min_leaf_samples;						// minimum samples in the leaves
	int n_classes;
	std::vector<node_t> tree;					// tree structure

	model_meta model;								// tree model (remove some useless information of tree structure)
	void check_param(int min_leaf_sample, int max_depth);
	void init(int min_leaf_samples, int max_depth);

	bool has_gen_model; 						// flag to indicate whether there is a model in the current class instance
public:
	BaseTree(int min_leaf_samples, int max_depth);
	~BaseTree();
	virtual void train(Dataset* ds, int max_feature, double *class_weight = NULL) = 0;
};

class DecisionTreeClassifier : public BaseTree {
protected:
	void build_tree(int max_feature, double *class_weight);
	void split(int max_feature, int *feature_list, double *class_weight,
				node_t *pa, node_t *lchilde, node_t *rchild);
	void gen_model();
	int* apply();
	double* predict(int* leaf_idx, bool is_proba);
public:
	DecisionTreeClassifier(int min_leaf_samples, int max_depth);
	
	// this method is for RandomForestClassifier, use a small fraction of features and do not copy training data into class
	void train(Dataset* ds, int max_feature, double *class_weight = NULL);

	void train(double *X, double *y, int sample_size, int feature_size, bool is_copy,
				int *discrete_idx = NULL, int discrete_size = 0, double *class_weight = NULL);
	void train(std::string filename, int feature_size, bool is_text,
				int *discrete_idx = NULL, int discrete_size = 0, double *class_weight = NULL);
	void train(std::string feature_filename, std::string label_filename, int feature_size,
				int *discrete_idx = NULL, int discrete_size = 0, double *class_weight = NULL);
	int* apply(double *X, int sample_size, int feature_size, bool is_copy);
	int* apply(std::string feature_filename, int feature_size, bool is_text);
	double* predict(double *X, int sample_size, int feature_size, bool is_proba = PROBA, bool is_copy = true);
	double* predict(std::string feature_filename, int feature_size, bool is_text, bool is_proba = PROBA);

	void dump(std::string model_filename);
	void load(std::string model_filename);
	void export_dotfile(std::string dot_filename);
	double* compute_importance();
};

class Criterion {
public:
	static double gini(double *arr, int size);
};

#endif

