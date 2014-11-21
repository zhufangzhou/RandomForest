/**
 * @file tree.cpp
 * @brief 
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-11-21
 */
#include "tree.h"

node::node(int n_classes) {
	this->is_cate = false;
	this->feature_id = -1;
	this->threshold = (feature_t)0.0;
	this->n_classes = n_classes;
	this->portion = new float[n_classes];	
	this->leaf_idx = -1; /* set to leaf node for default */
	this->left = this->right = NULL;
}

node::~node() {
	delete[] portion;
}

// ! TODO
void node::dump(const std::string& filename) {

}

// ! TODO
void node::dump(std::ofstream& ofs) {

}

tree::tree() {
			
}

tree::tree(std::string feature_rule, int max_depth, int min_split) {
	init(feature_rule, max_depth, min_split);
}

tree::~tree() {
	free_tree(this->root);	
	delete[] leaf_pt;
}

void tree::init(std::string feature_rule, int max_depth, int min_split) {
	this->feature_rule = feature_rule;
	this->max_depth = max_depth;
	this->min_split = min_split;
	this->leaf_pt = new node*[1];
	this->fea_imp = NULL;
}

void tree::free_tree(node*& nd) {
	if (nd->leaf_idx == -1) {
		delete nd;
	} else {
		free_tree(nd->left);
		free_tree(nd->right);
		delete nd;
	}
}

float* tree::compute_importance(bool re_compute) {
	std::stack<node*> st;
	node *c_node, *l_node, *r_node;
	if (fea_imp != NULL && !re_compute) {
		return fea_imp;
	} else {
		/* allocate memory to `fea_imp` */
		if (fea_imp != NULL) {
			fea_imp = (float*)realloc(fea_imp, sizeof(float)*n_features);
		} else {
			fea_imp = new float[n_features];
		}
		/* initialize to zero */
		memset(fea_imp, 0, sizeof(float)*n_features);
		st.push(root);	
		while (!st.empty()) {
			c_node = st.top();
			st.pop();

			if (c_node->leaf_idx != -1) {
				l_node = c_node->left;
				r_node = c_node->right;
				fea_imp[c_node->feature_id] += c_node->measure
						- 1.0*l_node->n_examples/c_node->n_examples*l_node->measure
						- 1.0*r_node->n_examples/c_node->n_examples*r_node->measure;
				st.push(r_node);
				st.push(l_node);
			}
		}
		float tot_imp = 0.0;
		for (int i = 0; i < n_features; i++) tot_imp += fea_imp[i];
		for (int i = 0; i < n_features; i++) fea_imp[i] /= tot_imp;
		return fea_imp;
	}
}
