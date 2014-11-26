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
	delete[] cur_frequency;
}

// ! TODO
void node::dump(const std::string& filename) {
	std::ofstream out(filename, std::ios::binary);
	dump(out);
	out.close();
}

// ! TODO
void node::dump(std::ofstream& out) {
	out.write((char*)&this->is_cate, sizeof(bool));
	out.write((char*)&this->feature_id, sizeof(int));
	out.write((char*)&this->threshold, sizeof(feature_t));
	out.write((char*)&this->gain, sizeof(float));
	out.write((char*)&this->n_classes, sizeof(int));
	out.write((char*)&this->leaf_idx, sizeof(int));
	out.write((char*)&this->cur_frequency, sizeof(float)*this->n_classes);
}

void node::print_info() {
	std::cout << std::endl
			  << "is_cate: " << std::boolalpha << this->is_cate << std::endl
			  << "feature_id: " << this->feature_id << std::endl
			  << "threshold: " << this->threshold << std::endl
			  << "gain: " << this->gain << std::endl
			  << "n_classes: " << this->n_classes << std::endl
			  << "leaf_idx: " << this->leaf_idx << std::endl;
	for (int c = 0; c < this->n_classes; c++)
		std::cout << this->cur_frequency[c] << " ";
	std::cout << std::endl << std::endl;
}

batch_node::batch_node(int n_classes) : node(n_classes) {

}

tree::tree() {
			
}

tree::tree(std::string feature_rule, int max_depth, int min_split) {
	init(feature_rule, max_depth, min_split);
}

tree::~tree() {
	free_tree(this->root);	
	delete[] leaf_pt;
	delete[] valid;
}

void tree::init(std::string feature_rule, int max_depth, int min_split) {
	this->feature_rule = feature_rule;
	this->max_depth = max_depth;
	this->min_split = min_split;
	this->leaf_pt = new node*[1];
	this->fea_imp = nullptr;
	this->valid = nullptr;

	/* set root node to nullptr */
	this->root = nullptr;
}

void tree::free_tree(node*& root) {
	if (root->leaf_idx == -1) {
		delete root;
	} else {
		free_tree(root->left);
		free_tree(root->right);
		delete root;
	}
}

float* tree::compute_importance(bool re_compute) {
	std::stack<node*> st;
	node *c_node, *l_node, *r_node;

	/* check if the tree has been built */
	if (this->root == nullptr) {
		std::cerr << "You need to build the tree before call this function" << std::endl;
		exit(EXIT_FAILURE);
	}

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
    node* c_node;
	int node_idx = 0, pa_idx;
	std::stack<node*> st;
	std::stack<int> st_idx;
	float tot_frequency;

	if (!ofs.is_open()) {
		std::cerr << "Cannot open file " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	// push root node to stack
	st.push(root);
	st_idx.push(-1);
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
				<< std::setprecision(3) << c_node->gain<< "\", shape=\"box\"];\n"
				<< std::endl;
			/* push left node and right node to stack */
			st.push(c_node->right);
			st.push(c_node->left);
			st_idx.push(node_idx);
			st_idx.push(node_idx);
        }
		node_idx++;
    }

	ofs << "}" << std::endl;
	ofs.close();
}

int tree::get_max_feature() {
	return this->max_feature;
}

decision_tree::decision_tree(const std::string feature_rule, int max_depth, int min_split) : tree(feature_rule, max_depth, min_split) {
	this->verbose = 0;
}

void decision_tree::build(dataset*& d) {
	target_t c; /* temporary variable to indicate current class */
	node* c_node;
	int n_classes = d->get_nclasses(), n_examples = d->get_nexamples(), n_features = d->get_nfeatures();
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
		this->max_feature == (int)log((double)n_features);
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

	ti->tic("Start build tree");	

	/* revursively build tree */
	build_rec(this->root, d, 0);

	ti->toc("Build tree done.");
}

void decision_tree::build_rec(node*& root, dataset*& d, int depth) {
	int n_classes = d->get_nclasses(), n_examples = d->get_nexamples(), count, tot_ex, left_tot_ex, right_tot_ex;
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
	if (depth >= this->max_depth || count < 2 || tot_ex <= this->min_split) {
		if (this->verbose > 0) {
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
		
		root->leaf_idx = 0;
		return;
	}

	/* 2. make a split */
	criterion* cr = new gini(root->cur_frequency, n_classes);
	s->split(this, root, d, cr);

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

	if (this->verbose > 0) {
		std::cout << "=================================" << std::endl;
		std::cout << "Depth: " << depth << std::endl;
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

	delete s;
	delete cr;
}

void decision_tree::debug(dataset*& d) {
	this->verbose = 1;
	build(d);	
	export_dotfile("tree.dot");
	std::cout << "Before dump" << std::endl;
	this->root->print_info();
	/* dump node */
	this->root->dump("root.model");
	std::ifstream in("root.model", std::ifstream::binary);
	batch_node* n = new batch_node(this->n_classes);	
	in.read((char*)&n->is_cate, sizeof(bool));
	in.read((char*)&n->feature_id, sizeof(int));
	in.read((char*)&n->threshold, sizeof(feature_t));
	in.read((char*)&n->gain, sizeof(float));
	in.read((char*)&n->n_classes, sizeof(int));
	in.read((char*)&n->leaf_idx, sizeof(int));
	in.read((char*)&n->cur_frequency, sizeof(float)*n->n_classes);
	std::cout << "After dump" << std::endl;
	n->print_info();
	delete n;
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
	delete[] left_frequency;
	delete[] right_frequency;
}

best_splitter::best_splitter(int n_classes) : splitter(n_classes) {

}

void best_splitter::split(tree* t, node*& root, dataset*& d, criterion*& cr) {
	ev_pair_t* x;
	int f, j, fst_valid, cur_ex, prev, prev_ex;
	float *zero_frequency, *nonzero_frequency; /* these two are for current node */
	float *left_frequency, threshold;
	int n_classes = d->get_nclasses(), max_feature = t->get_max_feature(); 

	/* the key idea of this sparse split is to determine where to put zero examples */
	zero_frequency = new float[n_classes];
	nonzero_frequency = new float[n_classes];

	/* did not set `right_frequency` because `current node` minus `left_frequency` is `right_frequency` */
	left_frequency = new float[n_classes];

	/* reset `criterion` class */
	cr->set_current(root->cur_frequency, n_classes);

	for (int i = 0; i < max_feature; i++) {
		f = i;		/* choose a feature to test */	
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

	delete[] zero_frequency;
	delete[] nonzero_frequency;
	delete[] left_frequency;
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

	delete[] right;
}

criterion::criterion() {
	is_init = false;
}

criterion::criterion(float*& frequency, int n_classes) {
	set_current(frequency, n_classes);
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
