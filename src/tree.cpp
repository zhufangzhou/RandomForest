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
	this->gain = 0.0;
	this->n_classes = n_classes;

	this->cur_frequency = new float[n_classes]();	

	this->leaf_idx = -1; /* set to leaf node for default */
	this->left = this->right = nullptr;
}

node::~node() {
	if (cur_frequency != nullptr) {
		delete[] cur_frequency;
		cur_frequency = nullptr;
	}
}

void node::dump(const std::string& filename) {
	std::ofstream out(filename, std::ios::binary);
	dump(out);
	out.close();
}

void node::dump(std::ofstream& out) {
	out.write((char*)&this->leaf_idx, sizeof(int));
	if (this->leaf_idx == -1) {
		out.write((char*)&this->is_cate, sizeof(bool));
		out.write((char*)&this->feature_id, sizeof(int));
		out.write((char*)&this->threshold, sizeof(feature_t));
		out.write((char*)&this->gain, sizeof(float));
		out.write((char*)&this->n_classes, sizeof(int));
	} else {
		out.write((char*)&this->n_classes, sizeof(int));
		out.write((char*)&this->cur_frequency, sizeof(float)*this->n_classes);
	}
}

void node::load(const std::string& filename) {
	std::ifstream in(filename, std::ios::binary);
	load(in);
	in.close();
}

void node::load(std::ifstream& in) {
	in.read((char*)&this->leaf_idx, sizeof(int));
	if (this->leaf_idx == -1) {
		in.read((char*)&this->is_cate, sizeof(bool));
		in.read((char*)&this->feature_id, sizeof(int));
		in.read((char*)&this->threshold, sizeof(feature_t));
		in.read((char*)&this->gain, sizeof(float));
		in.read((char*)&this->n_classes, sizeof(int));
	} else {
		in.read((char*)&this->n_classes, sizeof(int));
		in.read((char*)&this->cur_frequency, sizeof(float)*this->n_classes);
	}
}

void node::print_info() {
	float tot_frequency = 0.0;
	if (leaf_idx == -1) {
		std::cout << std::endl
				  << "*** Internal Node ***" << std::endl
				  << "is_cate: " << std::boolalpha << this->is_cate << std::endl
				  << "feature_id: " << this->feature_id << std::endl
				  << "threshold: " << this->threshold << std::endl
				  << "gain: " << this->gain << std::endl
				  << "n_classes: " << this->n_classes << std::endl
				  << "leaf_idx: " << this->leaf_idx << std::endl;
	} else {
		std::cout << std::endl
				  << "*** Leaf Node ***" << std::endl
				  << "n_classes: " << this->n_classes << std::endl;
		for (int c = 0; c < this->n_classes; c++) tot_frequency += this->cur_frequency[c];
		std::cout << "[ ";
		for (int c = 0; c < this->n_classes; c++) std::cout << this->cur_frequency[c] / tot_frequency << " ";
		std::cout << "]" << std::endl;
	}
}

batch_node::batch_node(int n_classes) : node(n_classes) {

}

tree::tree() {
			
}

tree::tree(std::string feature_rule, int max_depth, int min_split, int verbose) {
	init(feature_rule, max_depth, min_split, verbose);
}

tree::~tree() {
	free_tree(this->root);	
	if (leaf_pt != nullptr) {
		delete[] leaf_pt;
		leaf_pt = nullptr;
	}
	if (valid != nullptr) {
		delete[] valid;
		valid = nullptr;
	}
}

void tree::init(std::string feature_rule, int max_depth, int min_split, int verbose) {
	this->feature_rule = feature_rule;
	this->max_depth = max_depth;
	this->min_split = min_split;
	this->leaf_pt = new node*[1];
	this->leaf_size = 0;
	this->fea_imp = nullptr;
	this->valid = nullptr;
	this->verbose = verbose;

	/* set root node to nullptr */
	this->root = nullptr;
}

void tree::free_tree(node*& root) {
	if (root->leaf_idx == -1) {
		if (root != nullptr) {
			delete root;
			root = nullptr;
		}
	} else {
		free_tree(root->left);
		free_tree(root->right);
		if (root != nullptr) {
			delete root;
			root = nullptr;
		}
	}
}

void tree::check_build() {
	if (this->root == nullptr) {
		std::cerr << "You need to build the tree first!" << std::endl;
		exit(EXIT_FAILURE);
	}
}

//int* tree::apply(example_t* examples, int size) {
int* tree::apply(std::vector<example_t*> &examples) {
	feature_t *feature_vec;
	example_t *ex;
	int* ret, size = examples.size();
	node* cur_node;
	

	check_build();

	feature_vec = new feature_t[this->n_features]();
	ret = new int[size];

	for (int i = 0; i < size; i++) {
		ex = examples[i];
		for (int j = 0; j < ex->nnz; j++) feature_vec[ex->fea_id[j]] = ex->fea_value[j];
		cur_node = this->root;	
		/* go down the tree */
		while (cur_node->leaf_idx == -1) {
			if (cur_node->is_cate) { // split feature is categorical
				if (feature_vec[cur_node->feature_id] == cur_node->threshold) {
					cur_node = cur_node->left;
				} else {
					cur_node = cur_node->right;
				}
			} else { // split feature is continuous
				if (feature_vec[cur_node->feature_id] <= cur_node->threshold) {
					cur_node = cur_node->left;
				} else {
					cur_node = cur_node->right;
				}
			}
		}
		ret[i] = cur_node->leaf_idx;

		/* modify back */
		for (int j = 0; j < ex->nnz; j++) feature_vec[ex->fea_id[j]] = 0.0;
	}

	if (feature_vec != nullptr) {
		delete feature_vec;
		feature_vec = nullptr;
	}
	return ret;
}

int* tree::predict_label(std::vector<example_t*> &examples) {
	int *predict_leaf_idx, label, *ret, size = examples.size();
	node* leaf_node;
	float max_proba;

	/* let all the examples go down the tree */
	predict_leaf_idx = apply(examples);

	ret = new int[size];
	for (int i = 0; i < size; i++) {
		leaf_node = this->leaf_pt[predict_leaf_idx[i]];
		max_proba = 0.0;
		for (int c = 0; c < this->n_classes; c++) {
			if (leaf_node->cur_frequency[c] > max_proba) {
				max_proba = leaf_node->cur_frequency[c];
				label = c;
			}
		}
		ret[i] = label;
	}
	return ret;
}

/*
 * Return Vector Format Example:
 *
 * assume is a binary-classification
 * 				example1 		example2 		example3
 * class 0 		   0.8 			   0.9  		   0.3
 * class 1 		   0.2 			   0.1 			   0.7
 *
 * return [0.8, 0.9, 0.3, 0.2, 0.1, 0.7]
 */
float* tree::predict_proba(std::vector<example_t*> &examples) {
	int *predict_leaf_idx, size = examples.size();
	node* leaf_node;
	float* ret;

	/* let all the examples go down the tree */	
	predict_leaf_idx = apply(examples);

	ret = new float[size*this->n_classes];
	//for (int i = 1; i < size; i++) ret[i] = ret[0] + i*this->n_classes;
	
	for (int i = 0; i < size; i++) {
		leaf_node = this->leaf_pt[predict_leaf_idx[i]];
		for (int c = 0; c < this->n_classes; c++) {
			ret[i+size*c] = leaf_node->cur_frequency[c];
		}
		//memcpy(ret[i], leaf_node->cur_frequency, sizeof(float)*this->n_classes);
	}
	return ret;
}

float* tree::compute_importance(bool re_compute) {
	std::stack<node*> st;
	node *c_node, *l_node, *r_node;

	/* check if the tree has been built */
	check_build();

	if (fea_imp != nullptr && !re_compute) {
		return fea_imp;
	} else {
		/* allocate memory to `fea_imp` */
		if (fea_imp != nullptr) {
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

			if (c_node->leaf_idx == -1) {
				l_node = c_node->left;
				r_node = c_node->right;
				fea_imp[c_node->feature_id] += c_node->gain;
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

void tree::export_dotfile(const std::string& filename) {
	std::ofstream ofs(filename.c_str());
	int node_idx = 0;

	if (!ofs.is_open()) {
		std::cerr << "Cannot open file " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	export_dotfile(ofs, node_idx, true);

	/* close output stream */
	ofs.close();
}

void tree::export_dotfile(std::ofstream& ofs, int& node_idx, bool need_header_footer) {
    node* c_node;
	int pa_idx;
	std::stack<node*> st;
	std::stack<int> st_idx;
	float tot_frequency;

	if (!ofs.is_open()) {
		std::cerr << "You need to open the output stream first" << std::endl;
		exit(EXIT_FAILURE);
	}

	/* check the tree */
	check_build();

	// push root node to stack
	st.push(root);
	st_idx.push(-1);
	if (need_header_footer)
		ofs << "digraph Tree {" << std::endl;
	while (!st.empty()) {
		/* pop from stack */
        c_node = st.top();
		st.pop();
		pa_idx = st_idx.top();
		st_idx.pop();

		/* write relationship */
		if (pa_idx != -1) 
			ofs << pa_idx << " -> " << node_idx << ";" << std::endl;

		/* write node info */
        if (c_node->leaf_idx != -1) { /* leaf node */
            ofs << node_idx << " [label=\"" 
				<< "predict proba = [ ";
			tot_frequency = 0.0;
			for (int i = 0; i < c_node->n_classes; i++) tot_frequency += c_node->cur_frequency[i];
			for (int i = 0; i < c_node->n_classes; i++) {
				ofs << std::setprecision(3) << c_node->cur_frequency[i] / tot_frequency<< " ";
			}
			ofs << "]\", shape=\"box\"];" << std::endl;
        } else { /* internal node */
            if (c_node->is_cate) {
                ofs << node_idx << " [label=\"X[" << c_node->feature_id << "] = ";
            } else {
                ofs << node_idx << " [label=\"X[" << c_node->feature_id << "] <= ";
            }
			ofs	<< std::fixed << std::setprecision(3) << c_node->threshold << "\\ngain= "
				<< std::setprecision(3) << c_node->gain<< "\", shape=\"box\"];"
				<< std::endl;
			/* push left node and right node to stack */
			st.push(c_node->right);
			st.push(c_node->left);
			st_idx.push(node_idx);
			st_idx.push(node_idx);
        }
		node_idx++;
    }

	if (need_header_footer)
		ofs << "}" << std::endl;
}

int tree::add_leaf(node *leaf) {
	leaf_pt = (node**)realloc(this->leaf_pt, sizeof(node*)*(this->leaf_size+1));	
	leaf_pt[this->leaf_size] = leaf;
	this->leaf_size++;

	return this->leaf_size-1;
}

int tree::get_max_feature() {
	return this->max_feature;
}

int tree::get_n_features() {
	return this->n_features;
}

int tree::get_leaf_size() {
	return this->leaf_size;
}

decision_tree::decision_tree() {

}

// TODO
void decision_tree::dump(const std::string& filename) const {
	std::stack<node*> st;
	std::ofstream out(filename, std::ofstream::binary);	
	node *cur_node, *left_node, *right_node;

	out.write((char*)&this->n_classes, sizeof(int));
	out.write((char*)&this->n_features, sizeof(int));
	out.write((char*)&this->leaf_size, sizeof(int));

	if (this->root != nullptr) {
		this->root->dump(out);
		/* if has child, then push to stack */
		if (this->root->leaf_idx == -1)
			st.push(this->root);
	}

	while (!st.empty()){
		cur_node = st.top();
		st.pop();
		left_node = cur_node->left;
		right_node = cur_node->right;

		/* dump left node and right node */
		left_node->dump(out);
		right_node->dump(out);

		if (right_node->leaf_idx == -1) st.push(cur_node->right);
		if (left_node->leaf_idx == -1) st.push(cur_node->left);
	}

	out.close();
}

void decision_tree::load(const std::string& filename) {
	std::stack<node*> st;
	std::ifstream in(filename, std::ifstream::binary);
	node *cur_node, *left_node, *right_node;
	
	in.read((char*)&this->n_classes, sizeof(int));
	in.read((char*)&this->n_features, sizeof(int));
	in.read((char*)&this->leaf_size, sizeof(int));

	if (this->leaf_pt != nullptr) {
		delete[] this->leaf_pt;
		this->leaf_pt = nullptr;
	}
	// restore `leaf_pt`
	this->leaf_pt = new node*[this->leaf_size];

	this->root = new batch_node(this->n_classes);
	this->root->load(in);
	if (this->root->leaf_idx == -1) st.push(this->root);

	while (!st.empty()) {
		cur_node = st.top();
		st.pop();

		if (cur_node->leaf_idx == -1) {
			left_node = new batch_node(this->n_classes);
			left_node->load(in);
			right_node = new batch_node(this->n_classes);
			right_node->load(in);
			cur_node->left = left_node;
			cur_node->right = right_node;	

			/* push into stack */
			st.push(right_node);
			st.push(left_node);
		} else {
			/* rebuild `leaf_pt` */
			this->leaf_pt[cur_node->leaf_idx] = cur_node;
		}
	}
	in.clear();
}

decision_tree::decision_tree(const std::string feature_rule, int max_depth, int min_split, int verbose) : tree(feature_rule, max_depth, min_split, verbose) {

}

void decision_tree::build(dataset*& d) {
	target_t c; /* temporary variable to indicate current class */
	node* c_node;
	int n_classes = d->get_n_classes(), n_examples = d->get_n_examples(), n_features = d->get_n_features();
	int ex_id;
	float nf_t;
	m_timer* ti = new m_timer();

	/* set `n_features` and `n_classes` */
	this->n_features = n_features;
	this->n_classes = n_classes;

	/* determine `max_feature` */
	if (feature_rule == "sqrt") {
		this->max_feature = (int)sqrt((double)n_features);
	} else if (feature_rule == "log") {
		this->max_feature = (int)log((double)n_features);
	} else {
		nf_t = atof(feature_rule.c_str());	
		if (nf_t > 0.0 && nf_t <= 1) {
			this->max_feature = (int)(nf_t*n_features);
		} else if (nf_t > 1) {
			this->max_feature = std::min((int)nf_t, n_features);
		} else {
			std::cerr << "bad value of `feature_rule`: " << feature_rule 
				<< ", use `sqrt` as default" << std::endl;
			this->max_feature = (int)sqrt((double)n_features);
		}
	}

	this->valid = new int[n_examples];

	/* allocate space to root node */	
	root = new batch_node(n_classes);
	for (int i = 0; i < n_examples; i++) {
		/* TODO: Can add bootstrap here*/
		ex_id = i;

		c = d->y[ex_id];
		root->cur_frequency[c] += d->weight[c];

		/* set the chosen to be valid */
		this->valid[ex_id] = 1; 
	}

	if (verbose >= 1)
		ti->tic("Start build tree");	

	/* revursively build tree */
	build_rec(this->root, d, 0);

	if (verbose >= 1)
		ti->toc("Build tree done.");
}

void decision_tree::build_rec(node*& root, dataset*& d, int depth) {
	int n_classes = d->get_n_classes(), n_examples = d->get_n_examples(), count, tot_ex, left_tot_ex, right_tot_ex;
	splitter* s = new best_splitter(n_classes);			
	ev_pair_t *p;	
	node *first, *second;

	/* 1. check if reach to leaf */
	//if (depth >= this->max_depth) return; /* check depth */
	/* check purity and min split */
	count = tot_ex = 0;
	for (int c = 0; c < n_classes; c++) {
		tot_ex += root->cur_frequency[c] / d->weight[c];
		if (root->cur_frequency[c] >= 1e-5) {
			count++;
		}
	}
	if ((this->max_depth > 0 && depth >= this->max_depth) || count < 2 || tot_ex <= this->min_split) {
		if (this->verbose >= 2) {
			std::cout << "********************************" << std::endl;
			std::cout << "Depth: " << depth << std::endl;
			std::cout << "Different Class: " << count << std::endl;
			std::cout << "Total Example: " << tot_ex << std::endl;
			std::cout << "Valid Example: " << std::endl;
			for (int i = 0; i < n_examples; i++) {
				if (this->valid[i] > 0) {
					std::cout << "#" << i << ":" << d->y[i] << " ";
				}	
			}
			std::cout << std::endl << std::endl;
		}
		
		/* normalize */
		float tot_frequency = 0.0;
		for (int c = 0; c < root->n_classes; c++) tot_frequency += root->cur_frequency[c];
		for (int c = 0; c < root->n_classes; c++) root->cur_frequency[c] /= tot_frequency;

		/* attach this node to leaf node group */
		root->leaf_idx = add_leaf(root);
		return;
	}

	/* 2. make a split */
	criterion* cr = new gini(root->cur_frequency, n_classes);
	s->split(this, root, d, cr);
	
	// can't split any more
	if (s->fea_id == -1) {
		if (this->verbose >= 2) {
			std::cout << "********************************" << std::endl;
			std::cout << "Depth: " << depth << std::endl;
			std::cout << "Different Class: " << count << std::endl;
			std::cout << "Total Example: " << tot_ex << std::endl;
			std::cout << "Valid Example: " << std::endl;
			for (int i = 0; i < n_examples; i++) {
				if (this->valid[i] > 0) {
					std::cout << "#" << i << ":" << d->y[i] << " ";
				}	
			}
			std::cout << std::endl << std::endl;
		}
		
		/* normalize */
		float tot_frequency = 0.0;
		for (int c = 0; c < root->n_classes; c++) tot_frequency += root->cur_frequency[c];
		for (int c = 0; c < root->n_classes; c++) root->cur_frequency[c] /= tot_frequency;

		/* attach this node to leaf node group */
		root->leaf_idx = add_leaf(root);
		if (cr != nullptr) {
			delete cr;
			cr = nullptr;
		}
		return;
	}

	root->feature_id = s->fea_id;
	root->is_cate = d->is_cate[s->fea_id];
	root->threshold = s->threshold;
	root->gain = s->gain;
	root->n_classes = n_classes;
	root->leaf_idx = -1; /* indicate this node is not a leaf node */
	root->left = new batch_node(n_classes);
	memcpy(root->left->cur_frequency, s->left_frequency, sizeof(float)*n_classes);
	root->right = new batch_node(n_classes);
	memcpy(root->right->cur_frequency, s->right_frequency, sizeof(float)*n_classes);

	if (this->verbose >= 2) {
		std::cout << "=================================" << std::endl;
		std::cout << "Depth: " << depth << std::endl;
		std::cout << "Total Examples: " << tot_ex << std::endl;
		std::cout << "Split Feature: " << s->fea_id << " "
				  << "Threshold: " << s->threshold << std::endl;
		std::cout << "Valid Example: " << std::endl;
		for (int i = 0; i < n_examples; i++) {
			if (this->valid[i] > 0) {
				std::cout << "#" << i << ":" << d->y[i] << " ";
			}
		}
		std::cout << std::endl;
		std::cout << "Nonzero Values: " << std::endl;	
		for (int i = 0; i < d->size[s->fea_id]; i++) {
			if (this->valid[d->x[s->fea_id][i].ex_id] > 0)
				std::cout << d->x[s->fea_id][i].ex_id << ":"
					<< d->x[s->fea_id][i].fea_value << " ";
		}
		std::cout << std::endl << std::endl;
	}
	
	/* find the first example index in x[s->fea_id] which is large than threshold */
	p = d->x[s->fea_id];
	int l, k, u, m;
	k = 0;
	u = d->size[s->fea_id];
	while (k < u) {
		m = (k + u) / 2;	
		if (p[m].fea_value > s->threshold)
			u = m;
		else 
			k = m + 1;
	}
	/* build the node contains 0 examples first */
	if (s->threshold > 0) { /* 0s are in left */
		/* examples between l and u are in right */
		l = k;
		u = d->size[s->fea_id];
		first = root->left;
		second = root->right;
	} else {				/* 0s are in right */
		/* examples between l and u are in left */
		l = 0;
		u = k;
		first = root->right;
		second = root->left;
	}

	/* invalid second */
	for (int i = l; i < u; i++) 
		this->valid[p[i].ex_id] -= 1;	

	/* 3. build first node */
	build_rec(first, d, depth+1);	

	/* valid second(add one more to be decrease in the next for loop */
	for (int i = l; i < u; i++)
		this->valid[p[i].ex_id] += 2;
	/* all decrease 1 */
	for (int i = 0; i < n_examples; i++) 
		this->valid[i] -= 1;

	/* 4. build second node */
	build_rec(second, d, depth+1);
	
	/* restore valid */
	for (int i = 0; i < n_examples; i++)
		this->valid[i] += 1;
	for (int i = l; i < u; i++)
		this->valid[p[i].ex_id] -= 1;

	if (s != nullptr) {
		delete s;
		s = nullptr;
	}
	if (cr != nullptr) {
		delete cr;
		cr = nullptr;
	}
}


void decision_tree::debug(dataset*& d) {
	this->verbose = 1;
	build(d);	
	export_dotfile("tree.dot");
	/* dump tree */
	dump("tree.model");

	/* load tree */
	//decision_tree *t2 = new decision_tree();
	//t2->load("tree.model");
	//t2->export_dotfile("after.dot");
	//delete t2;
}

splitter::splitter(int n_classes) {
	left_frequency = new float[n_classes]();
	right_frequency = new float[n_classes]();
	this->gain = 0.0;
	this->fea_id = -1;
	this->threshold = 0.0;
	this->n_classes = n_classes;
}

splitter::~splitter() {
	if (left_frequency != nullptr) {
		delete[] left_frequency;
		left_frequency = nullptr;
	}
	if (right_frequency != nullptr) {
		delete[] right_frequency;
		right_frequency = nullptr;
	}
}

best_splitter::best_splitter(int n_classes) : splitter(n_classes) {

}

best_splitter::~best_splitter() {

}

void best_splitter::split(tree* t, node*& root, dataset*& d, criterion*& cr) {
	ev_pair_t* x;
	int f, j, fst_valid, cur_ex, prev, prev_ex;
	float *zero_frequency, *nonzero_frequency; /* these two are for current node */
	float *left_frequency, threshold;
	int n_classes = d->get_n_classes(), max_feature = t->get_max_feature(), n_features = t->get_n_features(); 

	/* the key idea of this sparse split is to determine where to put zero examples */
	zero_frequency = new float[n_classes];
	nonzero_frequency = new float[n_classes];

	/* did not set `right_frequency` because `current node` minus `left_frequency` is `right_frequency` */
	left_frequency = new float[n_classes];

	/* reset `criterion` class */
	cr->set_current(root->cur_frequency, n_classes);

	int* candidate_feature = new int[n_features], c_idx, tmp;
	for (int i = 0; i < n_features; i++) candidate_feature[i] = i;
	for (int i = 0; i < max_feature; i++) {
		//c_idx = m_random::getInstance().next_int(i, n_features);
		c_idx = d->valid_features[m_random::getInstance().next_int(i, d->n_valid)];
		tmp = candidate_feature[i]; 
		candidate_feature[i] = candidate_feature[c_idx]; 
		candidate_feature[c_idx] = tmp;
	}

	for (int i = 0; i < max_feature; i++) {
		f = candidate_feature[i];		/* choose a feature to test */	
		x = d->x[f];
		if (!d->is_cate[f]) { /* if the feature is continuous */
			/* find the first valid example index */
			fst_valid = -1; /* example index */
			for (j = 0; j < d->size[f]; j++) {
				cur_ex = x[j].ex_id;
				if (t->valid[cur_ex] > 0) { /* find an valid example */
					fst_valid = cur_ex;
					break;
				}
			}
			/* all the non-zero examples of this feature are not valid */
			if (fst_valid < 0) continue;

			/* here j mean the index of first valid example in `x`*/
			prev = j;

			/* Here we get two vector (`n_classes` dimensional), `zero_frequency` and `nonzero_frequency` */
			/* 1.get the frequency of nonzero examples */
			/* reset `nonzero_frequency` */
			memset(nonzero_frequency, 0, sizeof(float)*n_classes);
			for (j = prev; j < d->size[f]; j++) {
				cur_ex = x[j].ex_id;
				/* find all the valid example */
				if (t->valid[cur_ex] <= 0) continue;

				nonzero_frequency[d->y[cur_ex]] += d->weight[d->y[cur_ex]];
			}

			/* 2.except nonzero is zero */
			for (int c = 0; c < n_classes; c++) 
				zero_frequency[c] = root->cur_frequency[c] - nonzero_frequency[c]; 

			memset(left_frequency, 0, sizeof(float)*n_classes);
			/* if first valid example's feature value is positive, then zero examples must be in the left child node */		
			if (x[prev].fea_value > 0.0) {
				for (int c = 0; c < n_classes; c++) 
					left_frequency[c] += zero_frequency[c];
				
				/* as all nonzero feature value is positive, so the first split threshold should between 0 and x[prev].fea_value */
				threshold = 0.5*(0 + x[prev].fea_value);
				update(f, threshold, left_frequency, root, cr);
			}

			/* if first example's feature value is negative, then we search until x[prev].fea_value<0 && x[cur].fea_value>0 and put zero examples between them */
			/* `prev` means index in vector `x` */
			/* `prev_ex` means x[prev].ex_id */ 
			prev_ex = fst_valid;
			for (int cur = prev+1; cur < d->size[f]; cur++) {
				cur_ex = x[cur].ex_id;
				/* find all valid examples */
				if (t->valid[cur_ex] <= 0) continue;

				/* add current example to left */
				left_frequency[d->y[prev_ex]] += d->weight[d->y[prev_ex]];

				/* x[prev].fea_value        0        x[cur].fea_value */
				/*                     ^        ^                     */
				/* two thresholds to split denoted by ^ above */
				if (x[prev].fea_value < 0 && x[cur].fea_value > 0) {
					/* threshold 1 (x[prev].fea_value*/
					threshold = 0.5*(x[prev].fea_value + 0.0);	
					update(f, threshold, left_frequency, root, cr); 

					/* threshold 2 */
					/* add zero examples to left */
					for (int c = 0; c < n_classes; c++) 
						left_frequency[c] += zero_frequency[c];
					threshold = 0.5*(0.0 + x[cur].fea_value);
					update(f, threshold, left_frequency, root, cr);
				}

				/* test a split between x[prev].fea_value and x[cur].fea_value */
				if (x[prev].fea_value != x[cur].fea_value /* feature value of previous and current are different */
						&& d->y[prev_ex] != d->y[cur_ex] /* class label of previous and current are different */) {
					threshold = 0.5*(x[prev].fea_value + x[cur].fea_value);
					update(f, threshold, left_frequency, root, cr);
				}

				/* assign current info to prev */
				prev = cur;
				prev_ex = cur_ex;
			}

		} else { /* if the feature categorical */
			
		}
	}

	if (zero_frequency != nullptr) {
		delete[] zero_frequency;
		zero_frequency = nullptr;
	}
	if (nonzero_frequency != nullptr) {
		delete[] nonzero_frequency;
		nonzero_frequency = nullptr;
	}
	if (left_frequency != nullptr) {
		delete[] left_frequency;
		left_frequency = nullptr;
	}
	if (candidate_feature != nullptr) {
		delete candidate_feature;
		candidate_feature = nullptr;
	}
}

void best_splitter::update(int t_fea_id, float threshold, float*& left, node*& nd, criterion*& cr) {
	float* right = new float[n_classes]();
	for (int c = 0; c < n_classes; c++) 
		right[c] = nd->cur_frequency[c] - left[c];

	float t_gain;

	t_gain = cr->gain(left, right, n_classes);

	if (t_gain > this->gain) {
		this->gain = t_gain;
		this->fea_id = t_fea_id;
		this->threshold = threshold;
		memcpy(this->left_frequency, left, sizeof(float)*n_classes);
		memcpy(this->right_frequency, right, sizeof(float)*n_classes);
	}

	if (right != nullptr) {
		delete[] right;
		right = nullptr;
	}
}

criterion::criterion() {
	is_init = false;
}

criterion::criterion(float*& frequency, int n_classes) {
	set_current(frequency, n_classes);
}

criterion::~criterion() {

}

void criterion::set_current(float*& frequency, int n_classes) {
	this->cur_measure = measure(frequency, n_classes);
	this->cur_tot = this->tot_frequency;
	this->is_init = true;
}

float criterion::gain(float*& left_frequency, float*& right_frequency, int n_classes) {
	float left_tot, right_tot;
	float left_measure, right_measure;
	/* need to set current node before call gain function */
	if (!is_init) {
		std::cerr << "Please set current measure before call gain" << std::endl;
		exit(EXIT_FAILURE);
	}
	/* calculate left child node measure */
	left_measure = measure(left_frequency, n_classes);
	left_tot = this->tot_frequency;
	/* calculate right child node measure */
	right_measure = measure(right_frequency, n_classes);
	right_tot = this->tot_frequency;
	
	/* return gain value */
	return this->cur_measure 
		- (left_tot / this->cur_tot * left_measure)
		- (right_tot / this->cur_tot * left_measure);
}

gini::gini(float*& frequency, int n_classes) {
	set_current(frequency, n_classes);
}

gini::~gini() {

}

float gini::measure(float*& frequency, int n_classes) {
	float tot_frequency = 0.0, ret = 1.0;
	for (int c = 0; c < n_classes; c++) tot_frequency += frequency[c];
	for (int c = 0; c < n_classes; c++) {
		ret -= (frequency[c] / tot_frequency)*(frequency[c] / tot_frequency);
	}
	/* store the tot_frequency for other member funtion to use */
	this->tot_frequency = tot_frequency;

	return ret;
}
