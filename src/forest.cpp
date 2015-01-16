/**
 * @file forest.cpp
 * @brief 
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-12-21
 */
#include "forest.h"

forest::forest() {
	this->feature_rule = "sqrt";
	this->max_depth = -1;
	this->min_split = 1;
	this->n_trees = 10;
	this->n_threads = 1;

	fea_imp = nullptr;

	is_build = false;
}

forest::forest(const std::string feature_rule, int max_depth, int min_split, int n_trees, int n_threads, int verbose) {
	this->feature_rule = feature_rule;
	this->max_depth  = max_depth;
	this->min_split = min_split;
	this->n_trees = n_trees;
	this->n_threads = n_threads;
	this->verbose = verbose;

	fea_imp = nullptr;

	is_build = false;
}

forest::~forest() {
	free_forest();
}

void forest::free_forest() {
	/* free trees */
	for (int t = 0; t < this->trees.size(); t++) {
		if (this->trees[t] != nullptr)
			delete this->trees[t];
		this->trees[t] = nullptr;
	}
	this->trees.clear();

	/* free feature importance vector */
	if (fea_imp != nullptr) {
		delete[] fea_imp;
		fea_imp = nullptr;
	}
}

float* forest::compute_importance(bool re_compute) {
	float *tot_importance, *sub_importance;
	/* if has been computed before, just return */
	if (re_compute == false && this->fea_imp != nullptr) return this->fea_imp;

	if (this->fea_imp != nullptr) {
		delete[] this->fea_imp;
		this->fea_imp = nullptr;
	}

	tot_importance = new float[this->n_features]();
	
	for (int t = 0; t < this->n_trees; t++){
		/* compute feature importance of each tree estimator */
		sub_importance = this->trees[t]->compute_importance();
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
		threads[i] = std::thread([&, tree_begin, tree_end, ret]() {
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
		threads[i] = std::thread([&, tree_begin, tree_end, ret]() {
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

void forest::export_dotfile(const std::string& filename, dotfile_mode dm) {

	if (!check_build()) {
		std::cerr << "Please build tree before call `export_dotfile`." << std::endl;
		exit(EXIT_FAILURE);
	}

	if (dm == SEPARATE_TREES) {
		std::stringstream ss;
		std::string save_file_name;
		ss.str("");
		for (int t = 0; t < this->n_trees; t++) {
			ss << filename << (t+1);
			save_file_name = ss.str();
			this->trees[t]->export_dotfile(save_file_name);
			ss.str("");
		}
	} else {
		std::ofstream ofs(filename);
		if (!ofs.is_open()) {
			std::cerr << "Cannot open file " << filename << std::endl;
			exit(EXIT_FAILURE);
		}
		ofs << "digraph Forest {" << std::endl;
		int node_idx = 0;
		for (int t = 0; t < this->n_trees; t++) {
			this->trees[t]->export_dotfile(ofs, node_idx, false);
		}
		ofs << "}" << std::endl;

		ofs.clear();
	}
}

int* forest::get_leaf_counts() {
	int* ret;
	tree* c_tree;
	if (!check_build()) {
		std::cerr << "Please build the forest before getting `leaf_counts`" << std::endl;
		exit(EXIT_FAILURE);
	}

	ret = new int[this->n_trees];
	// collect leaf size for all trees
	for (int t = 0; t < this->n_trees; t++){
		c_tree = this->trees[t];
		ret[t] = c_tree->get_leaf_size();
	}
	return ret;
}

int forest::get_max_feature() {
	if (!check_build()) {
		std::cerr << "Please build the forest before getting `max_feature`" << std::endl;
		exit(EXIT_FAILURE);
	}
	return this->max_feature;
}

int forest::get_n_features() {
	if (!check_build()) {
		std::cerr << "Please build the forest before getting `n_features`" << std::endl;
		exit(EXIT_FAILURE);
	}
	return this->n_features;
}

int forest::get_n_classes() {
	if (!check_build()) {
		std::cerr << "Please build the forest before getting `n_classes`" << std::endl;
		exit(EXIT_FAILURE);
	}
	return this->n_classes;
}

bool forest::check_build() {
	return is_build;
}

random_forest_classifier::random_forest_classifier(const std::string feature_rule, int max_depth, int min_split, int n_trees, int n_threads, int verbose) : 
	forest(feature_rule, max_depth, min_split, n_trees, n_threads, verbose) {

}

random_forest_classifier::~random_forest_classifier() {
	free_forest();
}

void random_forest_classifier::parallel_build(int tree_begin, int tree_end, dataset*&d) {
	for (int t = tree_begin; t < tree_end; t++) {
		/* do not need any debug information to print during building process */
		this->trees[t] = new decision_tree(this->feature_rule, this->max_depth, this->min_split, 0);
		this->trees[t]->build(d);	
	}
}

void random_forest_classifier::build(dataset*& d) {
	int tree_begin, tree_end;
	m_timer* ti = new m_timer();

	if (verbose >= 1) 
		ti->tic("Starting build forest ...");	

	/* collect information from dataset */
	this->n_classes = d->get_n_classes();
	this->n_features = d->get_n_features();

	/* initialize trees */
	free_forest();
	this->trees.reserve(this->n_trees);

	/* parallel build tree */
	parallel_unit pu = init_block(this->n_trees, this->n_threads);
	std::vector<std::thread> threads(pu.num_threads - 1);

	tree_begin = 0;
	for (int i = 0; i < pu.num_threads - 1; i++) {
		tree_end = tree_begin + pu.block_size;
		threads[i] = std::thread([&, tree_begin, tree_end]() {
			parallel_build(tree_begin, tree_end, d);
		});
		tree_begin = tree_end;
	}
	parallel_build(tree_begin, this->n_trees, d);

	/* join all the threads */
	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

	/* collect max_feature after build */
	this->max_feature = this->trees[0]->get_max_feature();

	if (verbose >= 1)
		ti->toc("Done.");

	/* set the flag is_build to true */
	is_build = true;
}

void random_forest_classifier::dump(const std::string& filename) const {
	std::string save_file_name;
	std::stringstream ss;
	tree* c_tree;

	/* dump forest */
	std::ofstream out(filename+"0", std::ofstream::binary);
	// TODO: CAN ADD DUMP LEVEL HERE TO DECIDE HOW MUCH TO DUMP

	/* dump some parameters */
	out.write((char*)&this->n_trees, sizeof(int));
	out.write((char*)&this->n_threads, sizeof(int));
	out.write((char*)&this->n_classes, sizeof(int));
	out.write((char*)&this->n_features, sizeof(int));

	/* close file */
	out.close();

	/* dump trees separately */
	for (int t = 0; t < this->n_trees; t++) {
		/* tree suffix is start from 1, 0 is for forest */
		ss << filename << (t+1);
		save_file_name = ss.str();
		c_tree = this->trees[t];
		c_tree->dump(save_file_name);	
		ss.str("");
	}
}

void random_forest_classifier::load(const std::string& filename) {
	std::string load_file_name;
	std::stringstream ss;

	/* load forest */
	std::ifstream in(filename+"0", std::ifstream::binary);
	
	/* load forest level parameters */
	in.read((char*)&this->n_trees, sizeof(int));
	in.read((char*)&this->n_threads, sizeof(int));
	this->n_threads = 1; // !!!!!!!!!!!!!!!!!!!!!! fixed for debug
	in.read((char*)&this->n_classes, sizeof(int));
	in.read((char*)&this->n_features, sizeof(int));

	/* close file */
	in.close();

	/* allocate space for trees */
	this->trees.reserve(this->n_trees); 
	/* load trees separately */
	for (int t = 0; t < this->n_trees; t++) {
		/* tree suffix is start from 1, 0 is for forest */
		ss << filename << (t+1);
		load_file_name = ss.str();
		/* allocate space to each `tree` in the forest */
		this->trees[t] = new decision_tree();
		this->trees[t]->load(load_file_name);
		ss.str("");
	}

	/* set the `is_build` to true */
	this->is_build = true;
}

void random_forest_classifier::debug(dataset*& d) {
	
}
