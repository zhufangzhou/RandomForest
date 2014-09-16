#include "tree.h"

BaseTree::BaseTree(std::string filename, bool is_text, int feature_size, int min_leaf_samples, int max_depth, 
					int *discrete_idx, int discrete_size) {
	check_param(min_leaf_samples, max_depth, feature_size);
	init(min_leaf_samples, max_depth);
	if (is_text == TEXT) {
		this->ds->readText(filename, feature_size, discrete_idx, discrete_size);
	} else {
		this->ds->readBinary(filename, feature_size, discrete_idx, discrete_size);	
	}
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
	this->n_classes = 2;
}

void DecisionTreeClassifier::build_tree(int max_feature, double *class_weight) {
	node_t root, *cNode, lNode, rNode;
	int *feature_list = new int[max_feature];
	sample_size_t cNodeIdx = 0;
	
	// init root node
	root.depth = 1;					// root node's depth is set to 1
	// initialize the sample index
	root.sample_index = ordered_sequence<sample_size_t>(ds->sample_size);
	root.sample_size = ds->sample_size;
	// calculate weighted frequency 
	for (int i = 0; i < ds->sample_size; i++) {
		root.weighted_frequency[(int)ds->y[i]] += class_weight[(int)ds->y[i]];
	}
	// add root node to the tree structure vector
	tree.push_back(root);

	while (cNodeIdx < tree.size()) {
		// get a node to split
		cNode = &tree[cNodeIdx];

		if (cNode->depth == max_depth || cNode->sample_size < min_leaf_samples * 2) {	// cNode is leaf node
			cNode->is_leaf = true;
		} else {	// if cNode is not a leaf node , then split	into two children node
			// split the current and generate left and right child
			split(max_feature, feature_list, class_weight, cNode, &lNode, &rNode);

			
			// add left and right child to tree
			cNode->lchild = tree.size();
			tree.push_back(lNode);
			cNode->rchild = tree.size();
			tree.push_back(rNode);
		}
		
		// choose next node to split
		cNodeIdx++;
	}
}

void DecisionTreeClassifier::split(int max_feature, int *feature_list, double *class_weight, 
									node_t *pa, node_t *lchild, node_t *rchild) {
	int cFeature, bestFeature;						// current feature index and best feature index
	double criterionValue, bestCriterionValue = -1;	// here we choose gini index
	double proba, weighted_frequency_temp, lFraction, min_lr_gini_sum, lr_gini_sum;
	double *lWeighted_frequency, *rWeighted_frequency;
	discrete_t *dValues;
	int *dSample_Count;
	int dValue_count, *class_count;
	int *feature_order, sIdx, last_sIdx, last_oper_id, split_value = 0, left_size;

	lWeighted_frequency = new double[n_classes];
	rWeighted_frequency = new double[n_classes];
	// random sample `max_feature` features from the total feature set
	feature_list = random_sample(pa->sample_index, pa->sample_size, 1, max_feature, feature_list);
	for (int i = 0; i < max_feature; i++) {
		cFeature = feature_list[i];					// current feature index
		
		// parent node gini
		criterionValue = Criterion::gini(pa->weighted_frequency, n_classes);
		min_lr_gini_sum = 99999;					// a big real number (much larger than max gini)

		// deal with discrete and continuous feature (choose a best discrete value or choose a best split)
		if (ds->discrete_mask[cFeature] != -1) {	// discrete feature
			dValues = ds->discrete_value[ds->discrete_mask[cFeature]];		// distinct discrete values
			dValue_count = (int)dValues[0];									// distinct discrete value count
			dSample_Count = new int[(dValue_count + 1) * n_classes];		// each discrete value and each class contain sample size
			class_count = new int[n_classes];
			memset(dSample_Count, 0, sizeof(int)*(dValue_count+1)*n_classes);
			memset(class_count, 0, sizeof(int)*n_classes);
			// count sample size for each discrete value and each class
			for (int j = 0; j < pa->sample_size; j++) {
				sIdx = pa->sample_index[j];
				class_count[(int)ds->y[sIdx]]++;
				for (int v = 1; v <= dValue_count; v++) {
					if (ds->X[sIdx*ds->feature_size + cFeature] == dValues[v]) {
						dSample_Count[v*n_classes + (int)ds->y[sIdx]]++;
						break;
					}
				}
			}
			// test each discrete value and choose a best one
			int left_size_temp = 0;
			for (int v = 1; v <= dValue_count; v++) {
				left_size_temp = 0;
				for (int c = 0; c < n_classes; c++) {
					left_size_temp += dSample_Count[v*n_classes + c];
					lWeighted_frequency[c] = dSample_Count[v*n_classes + c] * class_weight[c];
					rWeighted_frequency[c] = (class_count[c] - dSample_Count[v*n_classes + c]) * class_weight[c];
				}
				lFraction = (double)left_size_temp / pa->sample_size;
				lr_gini_sum = lFraction*Criterion::gini(lWeighted_frequency, n_classes)
								+ (1-lFraction)*Criterion::gini(rWeighted_frequency, n_classes);
				// choose a best split point
				if (lr_gini_sum < min_lr_gini_sum) {
					min_lr_gini_sum = lr_gini_sum;
					split_value = dValues[v];
					left_size = left_size_temp;
				}
			}

			delete[] dSample_Count;
			delete[] class_count;
		} else {									// continuous feature
			// sort the active samples according to cFeature value
			feature_order = partial_argsort(ds->X, ds->sample_size, ds->feature_size, 
			pa->sample_index, pa->sample_size, cFeature, ASC, feature_order);
			
			// sample index = feature_order[j]
			last_sIdx = feature_order[0];

			// initialize weighted_frequency, left node is empty and all in right node
			memset(lWeighted_frequency, 0, sizeof(double)*n_classes);
			memcpy(rWeighted_frequency, pa->weighted_frequency, sizeof(double)*n_classes);
			
			last_oper_id = 0;
			
			for (int j = 1; j < pa->sample_size; j++) {
				sIdx = feature_order[j];
				if ((int)ds->y[sIdx] != (int)ds->y[last_sIdx]) {	// split when adjacent sample class is different
					// update left and right node's weighted frequency corresponding to this split
					weighted_frequency_temp = (j-last_oper_id)*class_weight[(int)ds->y[last_sIdx]];
					lWeighted_frequency[(int)ds->y[last_sIdx]] += weighted_frequency_temp;
					rWeighted_frequency[(int)ds->y[last_sIdx]] -= weighted_frequency_temp;
					// calculate gini sum of two child node
					lFraction = (double)j/pa->sample_size;
					lr_gini_sum = lFraction*Criterion::gini(lWeighted_frequency, n_classes)
								+ (1-lFraction)*Criterion::gini(rWeighted_frequency, n_classes);
					// choose a best split point
					if (lr_gini_sum < min_lr_gini_sum) {
						min_lr_gini_sum = lr_gini_sum;
						split_value = (ds->X[sIdx*ds->feature_size + cFeature] + ds->X[last_sIdx*ds->feature_size]) / 2;
						left_size = j;
					}
					last_oper_id = j;
				}

				last_sIdx = sIdx;
			}
		}

		criterionValue -= min_lr_gini_sum;
		// choose a best feature
		if (criterionValue > bestCriterionValue) {
			bestCriterionValue = criterionValue;
			pa->feature_index = cFeature;
			pa->feature_value = split_value;
			pa->is_discrete = (ds->discrete_mask[cFeature] != -1);
			lchild->sample_size = left_size;
			rchild->sample_size = pa->sample_size - left_size;
		}
	}
	lchild->sample_index = new int[lchild->sample_size];
	rchild->sample_index = new int[rchild->sample_size];
	// update left and right child node sample index using split information
	int l_count = 0, r_count = 0;
	for (int i = 0; i < pa->sample_size; i++) {
		sIdx = pa->sample_index[i];
		if ((ds->discrete_mask[pa->feature_index] == -1 && ds->X[sIdx*ds->feature_size + pa->feature_index] < pa->feature_value) 
			|| (ds->discrete_mask[pa->feature_index] != -1 && ds->X[sIdx*ds->feature_size + pa->feature_index] == pa->feature_value)){		// belong to left child
			lchild->sample_index[l_count++] = sIdx;
			lchild->weighted_frequency[(int)ds->y[sIdx]] += class_weight[(int)ds->y[sIdx]];
		} else {																		// belong to right child
			rchild->sample_index[r_count++] = sIdx;
			rchild->weighted_frequency[(int)ds->y[sIdx]] += class_weight[(int)ds->y[sIdx]];
		}
	}
	// set two child node's depth
	lchild->depth = rchild->depth = pa->depth + 1;

	// free parent node's sample index because it's no use
	delete[] pa->sample_index;

	// free space
	delete[] lWeighted_frequency;
	delete[] rWeighted_frequency;
}

double DecisionTreeClassifier::predict(double *feature_list){
	node_t *current_node = &this->tree[0];
	while(!current_node->is_leaf){
		if(current_node->is_discrete)
			if(feature_list[current_node->feature_index] == current_node->feature_value)
				current_node = &this->tree[current_node->lchild];
			else
				current_node = &this->tree[current_node->rchild];

		else
			if(feature_list[current_node->feature_index] < current_node->feature_value)
				current_node = &this->tree[current_node->lchild];
			else
				current_node = &this->tree[current_node->rchild];

	}
	return current_node->weighted_frequency[1] / (current_node->weighted_frequency[0] + current_node->weighted_frequency[1]);

}

double Criterion::gini(double *arr, int size) {
	double *proba = vec_normalize(arr, size, NOT_INPLACE);
	double gini = 1.0;
	for (int i = 0; i < size; i++) {
		gini -= proba[i] * proba[i];
	}
	delete[] proba;
	return gini;
}
