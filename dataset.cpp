/**
 * @file dataset.cpp
 * @brief 
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-11-19
 */
#include "dataset.h"

example_t::example_t() {
	nnz = 0;
	y = -1;
	fea_id = new int[1];
	fea_value = new feature_t[1];
}

example_t::~example_t() {
	delete[] fea_id;
	delete[] fea_value;
}

void example_t::push_back(int id, feature_t value) {
	fea_id = (int*)realloc(fea_id, sizeof(int)*(nnz+1));
	fea_value = (feature_t*)realloc(fea_value, sizeof(int)*(nnz+1));

	fea_id[nnz] = id;
	fea_value[nnz] = value;

	nnz++;
}

void example_t::debug() {
	if (y != -1) {
		std::cout << "Example Label: " << y << std::endl;
	}
	std::cout << "Features: " << std::endl;
	for (int i = 0; i < nnz; i++) {
		std::cout << fea_id[i] << ":" << fea_value[i] << " ";
	}
	std::cout << std::endl << std::endl;
}

data_reader::data_reader(const std::string& filename, int n_features, const learn_mode mode) {
	ifs.open(filename.c_str(), std::ios::binary);
	if (!ifs.is_open()) {
		std::cerr << "Can not open file " << filename << " ." << std::endl;
		exit(EXIT_FAILURE);
	}
	this->n_features = n_features;
	this->mode = mode;
}

data_reader::~data_reader() {
	if (ifs.is_open()) {
		ifs.close();
	}
}

example_t* data_reader::read_an_example() {
	example_t* ret;
	std::string line, t_str;
	int p_pos, c_pos, feature_id;
	feature_t feature_value;

	if (ifs.eof()) {
		return nullptr;
	}

	ret = new example_t();
	
	if (mode != PREDICT) {
		ifs >> ret->y;
		p_pos = 0; getline(ifs, line);
		c_pos = line.find(' ', 0);
	} else {
		p_pos = 0; c_pos = 0;
		getline(ifs, line);
	}

	if (line.length() < 1) return nullptr;
	
	while (p_pos <= c_pos) {
		p_pos = c_pos + 1;
		c_pos = line.find(':', p_pos);
		t_str = line.substr(p_pos, c_pos - p_pos);
		feature_id = atoi(t_str.c_str());

		p_pos = c_pos + 1;
		c_pos = line.find(' ', p_pos);
		feature_value = atof(line.substr(p_pos, c_pos - p_pos).c_str());

		if (feature_id >= n_features) {
			std::cerr << "input file feature id " << feature_id << " exceed `n_features` " << n_features << std::endl;
			exit(EXIT_FAILURE);
		}

		ret->push_back(feature_id, feature_value);
	}

	/* if read a blank line, just skip it */
	if (ret->nnz == 0) {
		return nullptr;
	}
	return ret;	
}

std::vector<example_t*> data_reader::read_examples() {
	example_t* single;
	std::vector<example_t*> ret;

	while( (single=read_an_example()) != nullptr) {
		ret.push_back(single);
	}

	return ret;
}

dataset::dataset() {
	is_init = false;
}

dataset::dataset(int n_classes, int n_features, float* weight) {
	init(n_classes, n_features, weight);
}

dataset::~dataset() {
	delete[] x;
	delete[] x[0];
	delete[] size;
	delete[] y;
	delete[] is_cate;	
}

void dataset::init(int n_classes, int n_features, float* weight) {
	this->n_classes = n_classes;
	this->n_features = n_features;
	this->x = new ev_pair_t*[this->n_features];
	this->y = nullptr;
	this->size = new int[this->n_features]();
	/* copy weight vector to dataset */
	this->weight = new float[this->n_classes];
	memcpy(this->weight, weight, sizeof(float)*this->n_classes);

	this->is_cate = new bool[this->n_features];
	this->is_init = true;
}

void dataset::load_data(const std::string& filename, const learn_mode mode) {
	data_reader* dr = new data_reader(filename, n_features, mode);
	std::vector<example_t*> ex_vec;
	/* te and tf correspond to each other */
	ev_pair_t* te; /* store ev_pair*/
	int* tf; /* store feature_id for sorting */
	int tot_size; /* total size in the dataset */
	m_timer* t = new m_timer();

	this->mode = mode;

	if (!is_init) {
		std::cerr << "Please init the dataset first" << std::endl;
		exit(EXIT_FAILURE);
	}

	/* read examples */
	t->tic("Loading data from file"+filename+" ...");
	ex_vec = dr->read_examples();
	n_examples = ex_vec.size();
	t->toc("Done.");

	/* generate dataset */
	t->tic("Generating dataset ...");
	te = new ev_pair_t[1];
	tf = new int[1];
	tot_size = 0;
	int ex_id = 0;
	/* predict mode does not need y arrary*/
	if (mode != PREDICT) {
		y = new target_t[1];
	}
	for (auto it = ex_vec.begin(); it != ex_vec.end(); it++, ex_id++) {
		/* allocate memory to variables */
		te = (ev_pair_t*)realloc(te, sizeof(ev_pair_t)*(tot_size+(*it)->nnz));
		tf = (int*)realloc(tf, sizeof(int)*(tot_size+(*it)->nnz));
		/* predict mode does not has label */
		if (mode != PREDICT) {
			y = (target_t*)realloc(y, sizeof(target_t)*(ex_id+1));
			/* check `y` between 0 ~ n_classes-1 */
			if ((*it)->y < 0 && (*it)->y >= n_classes) {
				std::cerr << "Label must between 0 and `n_classes`-1" << std::endl;
				exit(EXIT_FAILURE);
			}
			y[ex_id] = (*it)->y;
		}

		for (int i = 0; i < (*it)->nnz; i++) {
			size[(*it)->fea_id[i]]++;
			tf[tot_size] = (*it)->fea_id[i];
			te[tot_size].set(ex_id, (*it)->fea_value[i]);
			tot_size++;
		}
	}

	sort(te, tf, tot_size);
	int t_sum = 0;
	x[0] = te;
	for (int i = 1; i < n_features; i++) {
		t_sum += size[i-1];
		x[i] = x[0] + t_sum;	
	}
	t->toc("Done.");

	/* free space */
	delete[] tf;
}

void dataset::isort(ev_pair_t* a, int* f, int n){
    int i,j;
    float tv;
    int te;
    for(i=1; i<n; i++){
        for(j=i; j>0 && (f[j-1] > f[j] || (f[j-1] == f[j] && a[j-1].fea_value > a[j].fea_value)); j--){
            te=f[j];         	f[j]=f[j-1];                 		f[j-1]=te;
            te=a[j].ex_id; 		a[j].ex_id=a[j-1].ex_id; 			a[j-1].ex_id=te;
            tv=a[j].fea_value;  a[j].fea_value=a[j-1].fea_value;    a[j-1].fea_value=tv;
        }
    }
}

void dataset::qsortlazy(ev_pair_t* a, int* f, int l, int u){
    int i,j,r;
    float sv,tv;
    int se,te;
    if (u-l<7)
        return;
    r=l+rand()%(u-l);
    te=a[r].ex_id; 		a[r].ex_id=a[l].ex_id; 			a[l].ex_id=te;
    tv=a[r].fea_value;  a[r].fea_value=a[l].fea_value;  a[l].fea_value=tv;
    te=f[r];         	f[r]=f[l];                 		f[l]=te;
    i=l;
    j=u+1;
    while(1){
        do i++; while (i<=u && (f[i] < te || (f[i]==te && a[i].fea_value < tv)));
        do j--; while (f[j] > te || (f[j]==te && a[j].fea_value > tv));
        if (i>j)
            break;
        se=f[i];        	f[i]=f[j];           	    	 f[j]=se;
        se=a[i].ex_id; 		a[i].ex_id=a[j].ex_id; 			 a[j].ex_id=se;
        sv=a[i].fea_value;  a[i].fea_value=a[j].fea_value;   a[j].fea_value=sv;
    }
    te=a[l].ex_id; 		a[l].ex_id=a[j].ex_id; 			a[j].ex_id=te;
    tv=a[l].fea_value;  a[l].fea_value=a[j].fea_value;  a[j].fea_value=tv;
    te=f[l];         	f[l]=f[j];                 		f[j]=te;
    qsortlazy(a,f,l,j-1);
    qsortlazy(a,f,j+1,u);
}

void dataset::sort(ev_pair_t* a, int* f, int len){
    qsortlazy(a,f,0,len-1);
    isort(a,f,len);
}

int dataset::get_nclasses() {
	return this->n_classes;
}

int dataset::get_nexamples() {
	return this->n_examples;
}

int dataset::get_nfeatures() {
	return this->n_features;
}

void dataset::debug() {
	std::cout << "Class size: " << n_classes << std::endl;
	std::cout << "Example size: " << n_examples << std::endl;
	std::cout << "Feature size: " << n_features << std::endl;
	if (mode != PREDICT) {
		std::cout << "Labels: " << std::endl;
		for (int i = 0; i < n_examples; i++) {
			std::cout << y[i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "Features: " << std::endl;
	for (int i = 0; i < n_features; i++) {
		std::cout << "#" << i << "--> ";
		for (int j = 0; j < size[i]; j++) {
			std::cout << x[i][j].ex_id << ":" << x[i][j].fea_value << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
}
