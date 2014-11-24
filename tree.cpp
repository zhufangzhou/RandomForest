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

	this->cur_frequency = new float[n_classes]();	

	this->leaf_idx = -1; /* set to leaf node for default */
	this->left = this->right = nullptr;
}

node::~node() {
	delete[] cur_frequency;
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
	this->fea_imp = nullptr;

	/* initialize root node */
	this->root = nullptr;
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
            ofs << node_idx << " [label=\"gini = " 
				<< std::setprecision(3) << c_node->measure << "\\npositive proba = [ ";
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
			ofs	<< std::setprecision(3) << c_node->threshold << "\\ngini = "
				<< std::setprecision(3) << c_node->measure << "\", shape=\"box\"];\n"
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

void decision_tree::build(dataset*& d) {
	target_t c; /* temporary variable to indicate current class */
	std::stack<node*> st;
	node* c_node;
	/* allocate space to root node */	
	root = new batch_node(d->n_classes);
	for (int i = 0; i < d->n_examples; i++) {
		c = d->y[i];
		root->cur_frequency[c] += d->weight[i];
	}

	/* build tree */
	st.push(root);
	while (!st.empty()) {
		c_node = st.top();
		st.pop();

	}
}

splitter::splitter(int n_classes) {
	left_frequency = new float[n_classes]();
	right_frequency = new float[n_classes]();
}

splitter::~splitter() {
	delete[] left_frequency;
	delete[] right_frequency;
}

void best_splitter::split(tree*& t, node*& root, dataset*& d, criterion*& cr) {
	ev_pair_t* x;
	int j, fst_valid, cur_ex, prev, prev_ex;
	float *zero_frequency, *nonzero_frequency; /* these two are for current node */
	float *left_frequency, threshold;

	/* the key idea of this sparse split is to determine where to put zero examples */
	zero_frequency = new float[t->n_classes];
	nonzero_frequency = new float[t->n_classes];

	/* did not set `right_frequency` because `current node` minus `left_frequency` is `right_frequency` */
	left_frequency = new float[t->n_classes];

	/* reset `criterion` class */
	cr->set_current(root->cur_frequency, t->n_classes);

	for (int i = 0; i < t->max_feature; i++) {
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
			for (j = prev; j < d->size[f]; j++) {
				cur_ex = x[j].ex_id;
				/* find all the valid example */
				if (t->valid[cur_ex] <= 0) continue;

				/* reset `nonzero_frequency` */
				memset(nonzero_frequency, 0, sizeof(float)*t->n_classes);
				nonzero_frequency[d->y[cur_ex]] += d->weight[cur_ex];
			}

			/* 2.except nonzero is zero */
			for (int c = 0; c < t->n_classes; c++) 
				zero_frequency[c] = root->cur_frequency[c] - nonzero_frequency[c]; 

			memset(left_frequency, 0, sizeof(float)*t->n_classes);
			/* if first valid example's feature value is positive, then zero examples must be in the left child node */		
			if (x[prev].fea_value > 0.0) {
				for (int c = 0; c < t->n_classes; c++) 
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
				left_frequency[d->y[prev_ex]] += d->weight[prev_ex];

				/* x[prev].fea_value        0        x[cur].fea_value */
				/*                     ^        ^                     */
				/* two thresholds to split denoted by ^ above */
				if (x[prev].fea_value < 0 && x[cur].fea_value > 0) {
					/* threshold 1 (x[prev].fea_value*/
					threshold = 0.5*(x[prev].fea_value + 0.0);	
					update(f, threshold, left_frequency, root, cr); 

					/* threshold 2 */
					/* add zero examples to left */
					for (int c = 0; c < t->n_classes; c++) 
						left_frequency[c] += zero_frequency[c];
					threshold = 0.5*(0.0 + x[cur].fea_value);
					update(f, threshold, left_frequency, root, cr);
				}

				/* test a split between x[prev].fea_value and x[cur].fea_value */
				if (x[prev].fea_value != x[cur].fea_value /* feature value of previous and current are different */
						&& d->y[prev_ex] != d->y[cur_ex].fea_value /* class label of previous and current are different */) {
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

void best_splitter::update(int fea_id, float threshold, float* left, node*& nd, criterion*& cr) {
	float* right = new float[n_classes]();
	for (int c = 0; c < n_classes; c++) 
		right[c] = nd->cur_frequency[c] - left[c];

	float gain;

	gain = cr->gain(left, right, n_classes);

	if (gain > this->gain) {
		this->gain = gain;
		this->fea_id = fea_id;
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

criterion::set_current(float*& frequency, int n_classes) {
	this->measure = measure(frequency, n_classes);
	this->cur_tot = this->tot_frequency;
	this->is_init = true;
}

float criterion::gain(float*& cur_frequency, float*& left_frequency, float*& right_frequency, int n_classes) {
	float left_tot, right_tot;
	float left_measure, right_measure;
	if (!is_init) {
		std::cerr << "Please set current measure before call gain" << std::endl;
		exit(EXIT_FAILURE);
	}
	left_measure = measure(left_frequency, n_classes);
	left_tot = this->tot_frequency;
	right_measure = measure(right_frequency, n_classes);
	right_tot = this->tot_frequency;
	
	return this->cur_measure 
		- (left_tot/this->cur_tot*left_measure)
		- (right_tot/this->cur_tot*left_measure)
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
