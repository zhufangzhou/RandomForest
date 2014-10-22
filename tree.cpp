#include "tree.h"
#include "utils.h"
#include "dataset.h"

BaseTree::BaseTree(int min_leaf_samples, int max_depth) {
	if (max_depth == -1) max_depth = INF;
	check_param(min_leaf_samples, max_depth);
	init(min_leaf_samples, max_depth);
}

BaseTree::~BaseTree() {
	delete ds;
	delete[] model.tree;
	tree.clear();
}

void BaseTree::check_param(int min_leaf_samples, int max_depth) {
	if (min_leaf_samples <= 0) {
		std::cerr << "min_leaf_samples must be positive integer." << std::endl;
		exit(EXIT_FAILURE);
	}

	if (max_depth <= 0) {
		std::cerr << "max_depth must be positive integer." << std::endl;
		exit(EXIT_FAILURE);
	}
}

void BaseTree::init(int min_leaf_samples, int max_depth) {
	this->min_leaf_samples = min_leaf_samples;
	this->max_depth = max_depth;
	this->ds = new Dataset();
	this->n_classes = 2;
}

DecisionTreeClassifier::DecisionTreeClassifier(int min_leaf_samples, int max_depth) : BaseTree(min_leaf_samples, max_depth) {}

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
		cNode->criterion = Criterion::gini(cNode->weighted_frequency, n_classes);

		// if satisfy stopping condition or the node is pure
		if (cNode->depth == max_depth || cNode->sample_size < min_leaf_samples * 2 || 
			is_zero(cNode->weighted_frequency[0]) || is_zero(cNode->weighted_frequency[1])) {	// cNode is leaf node
			cNode->is_leaf = true;
			// delete leaf node's sample_index
			delete[] cNode->sample_index;
		} else {	// if cNode is not a leaf node , then split	into two children node
			// split the current and generate left and right child
			lNode.reset();
			rNode.reset();
			split(max_feature, feature_list, class_weight, cNode, &lNode, &rNode);

			// add left and right child to tree
			cNode->lchild = tree.size();
			cNode->rchild = tree.size()+1;
			tree.push_back(lNode);
			tree.push_back(rNode);
		}
		
		// choose next node to split
		cNodeIdx++;
	}

	delete[] feature_list;
}

void DecisionTreeClassifier::split(int max_feature, int *feature_list, double *class_weight, 
									node_t *pa, node_t *lchild, node_t *rchild) {
	int cFeature, bestFeature;						// current feature index and best feature index
	double criterionValue, bestCriterionValue = -1;	// here we choose gini index
	double proba, weighted_frequency_temp, lFraction, min_lr_gini_sum, lr_gini_sum, split_value = 0;
	double *lWeighted_frequency, *rWeighted_frequency;
	discrete_t *dValues;
	int *dSample_Count;
	int dValue_count, *class_count;
	int *feature_order, sIdx, last_sIdx, last_oper_id, left_size;

	lWeighted_frequency = new double[n_classes];
	rWeighted_frequency = new double[n_classes];
	feature_order = new int[pa->sample_size];
	// random sample `max_feature` features from the total feature set
	feature_list = random_sample(ds->feature_size, max_feature, feature_list);
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
				// split when adjacent sample class and feature value is different
				if ( ((int)ds->y[sIdx] != (int)ds->y[last_sIdx]) &&
					 (ds->X[sIdx*ds->feature_size + cFeature] != ds->X[last_sIdx*ds->feature_size + cFeature]) ) {	
					// update left and right node's weighted frequency corresponding to this split
					int idx_temp;
					for (int k = last_oper_id; k < j; k++) {
						idx_temp = feature_order[k];
						weighted_frequency_temp = class_weight[(int)ds->y[idx_temp]];
						lWeighted_frequency[(int)ds->y[idx_temp]] += weighted_frequency_temp;
						rWeighted_frequency[(int)ds->y[idx_temp]] -= weighted_frequency_temp;
					}
					// calculate gini sum of two child node
					lFraction = (double)j/pa->sample_size;
					lr_gini_sum = lFraction*Criterion::gini(lWeighted_frequency, n_classes)
								+ (1-lFraction)*Criterion::gini(rWeighted_frequency, n_classes);
					// choose a best split point
					if (lr_gini_sum < min_lr_gini_sum) {
						min_lr_gini_sum = lr_gini_sum;
						split_value = (ds->X[sIdx*ds->feature_size + cFeature] + ds->X[last_sIdx*ds->feature_size + cFeature]) / 2;
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
			pa->improvement = bestCriterionValue;
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
	delete[] feature_order;
}

void DecisionTreeClassifier::train(double *X, double *y, int sample_size, int feature_size, 
					int *discrete_idx, int discrete_size, double *class_weight) {
	// if not specify class_weight, use ones vector as default
	if (class_weight == NULL) {
		class_weight = new double[n_classes];
		for (int i = 0; i < n_classes; i++) class_weight[i] = 1.0;
	}
	ds->set_dataset(X, y, sample_size, feature_size, discrete_idx, discrete_size);

	// train model
	build_tree(feature_size, class_weight);
	// generate model using tree structure
	gen_model();
}

void DecisionTreeClassifier::train(std::string filename, int feature_size, bool is_text, 
					int *discrete_idx, int discrete_size, double *class_weight) {
	// if not specify class_weight, use ones vector as default
	if (class_weight == NULL) {
		class_weight = new double[n_classes];
		for (int i = 0; i < n_classes; i++) class_weight[i] = 1.0;
	}
	// read data from file
	if (is_text) {
		ds->readText(filename, feature_size, TRAIN, discrete_idx, discrete_size);
	} else {
		ds->readBinary(filename, feature_size, TRAIN, discrete_idx, discrete_size);
	}

	// train model
	build_tree(feature_size, class_weight);
	// generate model using tree structure
	gen_model();
}

void DecisionTreeClassifier::train(std::string feature_filename, std::string label_filename, int feature_size,
					int *discrete_idx, int discrete_size, double *class_weight) {
	// if not specify class_weight, use ones vector as default
	if (class_weight == NULL) {
		class_weight = new double[n_classes];
		for (int i = 0; i < n_classes; i++) class_weight[i] = 1.0;
	}
	// read data from feature file and label file (only for binary file)
	ds->readBinary(feature_filename, label_filename, feature_size, discrete_idx, discrete_size);

	// train model
	build_tree(feature_size, class_weight);
	// generate model using tree structure
	gen_model();
}

void DecisionTreeClassifier::gen_model() {
	double node_value;
	model.tree = new model_t[tree.size()];
	int counts = 0;
	for (auto it = tree.begin(); it != tree.end(); it++) {
		node_value = it->weighted_frequency[1] / (it->weighted_frequency[0] + it->weighted_frequency[1]);
		model.tree[counts++] = model_t(it->is_leaf, node_value, it->feature_index, it->is_discrete, it->feature_value,
			it->improvement, it->criterion, it->depth, it->lchild, it->rchild);
	}
	model.model_size = counts;
	model.feature_size = ds->feature_size;
}

int* DecisionTreeClassifier::apply() {
	model_t *cNode;
	int *leaf_idx, cIdx, feature_idx;
	double cFeature_value, threshold;
	leaf_idx = new int[ds->sample_size];
	
	for (int i = 0; i < ds->sample_size; i++) {
		cIdx = 0;
		cNode = &model.tree[cIdx];
		while (!cNode->is_leaf) {
			threshold = cNode->feature_value;
			feature_idx = cNode->feature_index;
			cFeature_value = ds->X[i*ds->feature_size + feature_idx];
			// whether is discrete feature or not 
			if (cNode->is_discrete) {
				if (cFeature_value == threshold) {
					cIdx = cNode->lchild;
				} else {
					cIdx = cNode->rchild;
				}
			} else {
				if (cFeature_value < threshold) {
					cIdx = cNode->lchild;
				} else {
					cIdx = cNode->rchild;
				}
			}
			cNode = &model.tree[cIdx];
		}
		leaf_idx[i] = cIdx;
	}
	return leaf_idx;
}

int* DecisionTreeClassifier::apply(std::string filename, int feature_size, bool is_text) {
	// read data from file
	if (is_text) {
		ds->readText(filename, feature_size, PREDICT);
	}
	else {
		ds->readBinary(filename, feature_size, PREDICT);
	}

	return apply();
}

int* DecisionTreeClassifier::apply(double *X, int sample_size, int feature_size) {
	// set the dataset
	ds->set_dataset(X, sample_size, feature_size);

	return apply();
}

double* DecisionTreeClassifier::predict(int *leaf_idx, bool is_proba) {
	double *ret;
	ret = new double[ds->sample_size];
	if (is_proba) {
		for (int i = 0; i < ds->sample_size; i++) {
			ret[i] = model.tree[leaf_idx[i]].node_value;
		}
	}
	else {
		for (int i = 0; i < ds->sample_size; i++) {
			ret[i] = (int)(model.tree[leaf_idx[i]].node_value + 0.5);
		}
	}
	delete[] leaf_idx;
	return ret;
}

double* DecisionTreeClassifier::predict(double *X, int sample_size, int feature_size, bool is_proba) {
	int *leaf_idx;
	
	leaf_idx = apply(X, sample_size, feature_size);
	if (leaf_idx == NULL) {
		std::cerr << "Fail to predict." << std::endl;
		exit(EXIT_FAILURE);
	}

	return predict(leaf_idx, is_proba);
}

double* DecisionTreeClassifier::predict(std::string filename, int feature_size, bool is_text, bool is_proba){
	int *leaf_idx;
	leaf_idx = apply(filename, feature_size, is_text);
	if (leaf_idx == NULL) {
		std::cerr << "Fail to predict." << std::endl;
		exit(EXIT_FAILURE);
	}
	
	return predict(leaf_idx, is_proba);
}

double* DecisionTreeClassifier::compute_importance() {
	double *importance;
	model_t cNode;
	if (model.tree == NULL) {
		std::cerr << "Please call `train` function first." << std::endl;
		exit(EXIT_FAILURE);
	}
	importance = new double[model.feature_size];
	memset(importance, 0, sizeof(double) * model.feature_size);
	// add all the improvement from those node who use this feature as a split feature
	for (int i = 0; i < model.model_size; i++) {
		cNode = model.tree[i];
		if (!cNode.is_leaf) {
			importance[cNode.feature_index] += cNode.improvement;
		}
	}
	// normalize the importance
	vec_normalize(importance, model.feature_size, INPLACE);
	return importance;
}

void DecisionTreeClassifier::dump(std::string model_filename) {
	FILE *fp;
	if ((fp = fopen(model_filename.c_str(), "wb")) == NULL) {
		std::cerr << "Fail to open the model file: " << model_filename << "." << std::endl;
		exit(EXIT_FAILURE);
	}
	if (model.tree == NULL) {
		std::cerr << "Please call `train` function first." << std::endl;
		exit(EXIT_FAILURE);
	}

	timer.tic("Begin Dump Model.");
	// dump the model to model_filename
	fwrite(&model.model_size, sizeof(int), 1, fp);
	fwrite(&model.feature_size, sizeof(int), 1, fp);
	fwrite(model.tree, sizeof(model_t), model.model_size, fp);
	timer.toc("Finish Dump Model.");
}

void DecisionTreeClassifier::load(std::string model_filename) {
	FILE *fp;
	if ((fp = fopen(model_filename.c_str(), "wb")) == NULL) {
		std::cerr << "Fail to open the model file: " << model_filename << "." << std::endl;
		exit(EXIT_FAILURE);
	}
	// delete current model first
	if (model.tree != NULL)
		delete[] model.tree;

	timer.tic("Begin Load Moedl.");
	fread(&model.model_size, sizeof(int), 1, fp);
	fread(&model.feature_size, sizeof(int), 1, fp);
	fread(model.tree, sizeof(model_t), model.model_size, fp);
	timer.toc("Finish Load Model.");
}

void DecisionTreeClassifier::export_dotfile(std::string dot_filename) {
	FILE *fp;
	model_t cNode;
	if (model.tree == NULL) {
		std::cerr << "Please call `train` function first." << std::endl;
		exit(EXIT_FAILURE);
	}
	fp = fopen(dot_filename.c_str(), "w");
	fprintf(fp, "digraph Tree {\n");
	for (int i = 0; i < model.model_size; i++) {
		cNode = model.tree[i];
		if (cNode.is_leaf) {
			fprintf(fp, "%d [label=\"gini = %.3lf\\npositive proba = %.3lf\", shape=\"box\"];\n", i, cNode.criterion, cNode.node_value);
		} else {
			if (cNode.is_discrete) {
				fprintf(fp, "%d [label=\"X[%d] = %.3lf\\ngini = %.3lf\", shape=\"box\"];\n", i, cNode.feature_index, cNode.feature_value, cNode.criterion);
			} else {
				fprintf(fp, "%d [label=\"X[%d] < %.3lf\\ngini = %.3lf\", shape=\"box\"];\n", i, cNode.feature_index, cNode.feature_value, cNode.criterion);
			}
			fprintf(fp, "%d -> %d;\n", i, cNode.lchild);
			fprintf(fp, "%d -> %d;\n", i, cNode.rchild);
		}
	}
	fprintf(fp, "}\n");
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

