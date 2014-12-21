/**
 * @file forest.cpp
 * @brief 
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-12-21
 */
#include "forest.h"

float* forest::compute_importance(bool re_compute = false) {
	float *tot_importance, *sub_importance;
	tot_importance = new float[this->n_features]();
	
	for (auto t = this->trees.begin(); t != this->trees.end(); t++){
		/* compute feature importance of each tree estimator */
		sub_importance = t->compute_importance();
		for (int i = 0; i < this->n_features; i++) {
			tot_importance[i] += sub_importance[i] / this->n_trees;
		}
	}

	/* save the pointer */
	this->fea_imp = tot_importance;
	return tot_importance;
}

float* forest::predict_proba(std::vector<example_t*> &examples) {
	int example_size;
	float *ret, *sub_proba;

	example_size = examples.size();
	ret = new float[example_size*this->n_classes]();
	for (auto t = this->tree.begin; t != this->trees.end(); t++) {
		sub_proba = t->predict_proba(examples);	
	}
}

random_forest_classifier::random_forest_classifier(const std::stirng feature_rule, int max_depth, int min_split) {

}
