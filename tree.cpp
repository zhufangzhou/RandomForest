#include "tree.h"

BaseTree::BaseTree(std::string filename, int feature_size, int min_leaf_samples, int max_depth) {
	check_param(min_leaf_samples, max_depth, feature_size);
	init(min_leaf_samples, max_depth);
	this->ds->readBinary(filename, feature_size);
}

BaseTree::BaseTree(std::string filename, int feature_size, int min_leaf_samples, int max_depth) {
	check_param(min_leaf_samples, max_depth, 1);
	init(min_leaf_samples, max_depth);
	this->ds->readText(filename, feature_size);
}

void BaseTree::check_param(int min_leaf_samples, int max_depth, int feature_size) {
	if (min_leaf_samples <= 0) {
		std::cout << "min_leaf_samples must be positive integer." << std::endl;
		exit(EXIT_FAILURE);
	}

	if (max_depth <= 0) {
		std::cout << "max_depth must be positive integer." << std::endl;
		exit(EXIT_FAILURE);
	}

	if (feature_size <= 0) {
		std::cout << "feature_size must be positive integer." << std::endl;
		exit(EXIT_FAILURE);
	}
}

void BaseTree::init(int min_leaf_samples, int max_depth) {
	this->min_leaf_samples = min_leaf_samples;
	this->max_depth = max_depth;
	this->ds = new Dataset();
}

void DecisionTreeClassifier::build_tree(int max_feature) {
	node_t root, *cNode, lNode, rNode;
	class Splitter mSplitter;
	sample_size_t cNodeIdx = 0;
	
	// root node's depth is set to 1
	root.depth = 1;
	// initialize the sample index
	root.sample_index = ordered_sequence<sample_size_t>(ds->sample_size);
	tree.push_back(root);

	while (cNodeIdx < tree.size()) {
		// get a node to split
		cNode = &tree[cNodeIdx];

		if (cNode->depth == max_depth || cNode->sample_size <= min_leaf_samples) {
			cNode->is_leaf = true;
		} else {	// split	
			lNode_sample_size = rNode_sample_size = 0;
			// split the current and generate left and right child
			mSplitter.split(ds->feature_size, max_feature, cNode, &lNode, &rNode);

			// free parent node's sample index because it's no use
			if (cNode->sample_index != NULL)
				delete[] cNode->sample_index;

			// add left and right child to tree
			tree.push_back(lNode);
			tree.push_back(rNode);
		}
		
		// choose next node to split
		cNodeIdx++;
	}
}