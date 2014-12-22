/**
 * @file forest.cpp
 * @brief 
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-12-21
 */
#include "forest.h"

forest::forest() {

}

forest::~forest() {
	free_forest();
}

void forest::free_forest() {

}

float* forest::compute_importance(bool re_compute) {
	float *tot_importance, *sub_importance;
	tot_importance = new float[this->n_features]();
	
	for (auto t = this->trees.begin(); t != this->trees.end(); t++){
		/* compute feature importance of each tree estimator */
		sub_importance = (*t)->compute_importance();
		for (int i = 0; i < this->n_features; i++) {
			tot_importance[i] += sub_importance[i] / this->n_trees;
		}
	}

	/* save the pointer */
	this->fea_imp = tot_importance;
	return tot_importance;
}

void forest::parallel_predict_proba(int tree_begin, int tree_end, std::vector<example_t*> &examples, float* ret) {
	tree* cur_tree;
	float* sub_proba;
	int example_size = examples.size();

	for (int t = tree_begin; t < tree_end; t++) {
		cur_tree = trees[t];	
		sub_proba = cur_tree->predict_proba(examples);	
		for (int i = 0; i < example_size * this->n_classes; i++) {
			ret[i] += sub_proba[i];
		}
		delete[] sub_proba;
	}
}

/*
 * Return Vector Format Example:
 *
 * assume is a binary-classification
 * 					example1 			example2 			example3
 * class 0 			  0.8 				   0.9 				   0.3
 * class 1 			  0.2 				   0.1 				   0.7
 *
 * return [0.8, 0.9, 0.3, 0.2, 0.1, 0.7]
 */
float* forest::predict_proba(std::vector<example_t*> &examples) {
	int example_size, tree_begin, tree_end;
	float *ret; 

	example_size = examples.size();
	ret = new float[example_size*this->n_classes]();

	// init the parallel unit
	parallel_unit pu = init_block(this->n_trees, this->n_threads);
	std::vector<std::thread> threads(pu.num_threads - 1);

	tree_begin = 0;
	for (int i = 0; i < pu.num_threads - 1; i++) {
		// calculate the tree_begin and tree_end
		tree_end = tree_begin + pu.block_size;
		threads[i] = std::thread([&]() {
				parallel_predict_proba(tree_begin, tree_end, examples, ret);
		});
		tree_begin = tree_end;
	}

	// do last set of trees in this thread
	parallel_predict_proba(tree_begin, this->n_trees, examples, ret);

	// join all the threads
	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

	// normalize
	for (int i = 0; i < example_size * this->n_classes; i++) {
		ret[i] /= this->n_trees;
	}
	
	return ret;
}

int* forest::predict_label(std::vector<example_t*> &examples) {
	float *proba, max_proba;
	int example_size, *ret;

	proba = predict_proba(examples);
	example_size = examples.size();

	for (int i = 0; i < example_size; i++) {
		max_proba = 0.0;
		for (int c = 0; c < this->n_classes; c++) {
			// find a label with maximum probability
			if (proba[i+example_size*c] > max_proba) {
				max_proba = proba[i+example_size*c];
				ret[i] = c;
			}
		}
	}

	delete[] proba;
	return ret;
}

/*
 * Return Vector Format Example:
 * 					Tree1 			Tree2 			Tree3
 * Example1 		  0 			  5 			  9
 * Example2 		  8 			  6 			  2
 * Example3 		  3 			  10 			  1
 *
 * return [0, 5, 9, 8, 6, 2, 3, 10, 1]
 */
void forest::parallel_apply(int tree_begin, int tree_end, std::vector<example_t*> &examples, int* ret) {
	tree* c_tree;
	int *sub_idx, example_size = examples.size();

	for (int t = tree_begin; t < tree_end; t++) {
		c_tree = this->trees[t];	
		sub_idx = c_tree->apply(examples);
		for (int i = 0; i < example_size; i++) {
			ret[t+i*this->n_trees] = sub_idx[i];
		}
	}
}

// for each given example, return a leaf index which it lies in each tree
int* forest::apply(std::vector<example_t*> &examples) {
	int tree_begin, tree_end, example_size = examples.size();
	int* ret;

	ret = new int[example_size * this->n_trees]();

	parallel_unit pu = init_block(this->n_trees, this->n_threads);
	std::vector<std::thread> threads(pu.num_threads - 1);

	tree_begin = 0;
	for (int i = 0; i < pu.num_threads - 1; i++) {
		tree_end = tree_begin + pu.block_size;
		threads[i] = std::thread([&]() {
			parallel_apply(tree_begin, tree_end, examples, ret);		
		});
		tree_begin = tree_end;
	}
	// do the last piece
	parallel_apply(tree_begin, this->n_trees, examples, ret);

	// join all the threads
	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

	return ret;
}

int* forest::get_leaf_counts() {
	int* ret;
	tree* c_tree;

	ret = new int[this->n_trees];
	// collect leaf size for all trees
	for (int t = 0; t < this->n_trees; t++){
		c_tree = this->trees[t];
		ret[t] = c_tree->get_leaf_size();
	}
	return ret;
}

random_forest_classifier::random_forest_classifier(const std::string feature_rule, int max_depth, int min_split) {

}
