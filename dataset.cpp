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
	fea_id = (int*)realloc(fea_id, nnz+1);
	fea_value = (feature_t*)realloc(fea_value, nnz+1);

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

DataReader::DataReader(const std::string& filename, int n_features, const learn_mode mode) {
	ifs.open(filename.c_str(), std::ios::binary);
	if (!ifs.is_open()) {
		std::cerr << "Can not open file " << filename << " ." << std::endl;
		exit(EXIT_FAILURE);
	}
	this->n_features = n_features;
	this->mode = mode;
}

DataReader::~DataReader() {
	if (ifs.is_open()) {
		ifs.close();
	}
}

example_t* DataReader::read_an_example() {
	example_t* ret;
	std::string line, t_str;
	int p_pos, c_pos, feature_id;
	feature_t feature_value;

	if (ifs.eof()) {
		return NULL;
	}

	ret = new example_t();
	
	if (mode == TRAIN) {
		ifs >> ret->y;
		p_pos = 0;
		getline(ifs, line);
		c_pos = line.find(' ', 0);
	} else {
		p_pos = 0;
		c_pos = 0;
		getline(ifs, line);
	}
	
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
		return NULL;
	}
	return ret;	
}

std::vector<example_t*> DataReader::read_examples() {
	example_t* single;
	std::vector<example_t*> ret;

	while( (single=read_an_example()) != NULL ) {
		ret.push_back(single);
	}

	return ret;
}

Dataset::Dataset() {
	is_init = false;
}

Dataset::Dataset(int n_classes, int n_features) {
	init(n_classes, n_features);
}

Dataset::~Dataset() {
	delete[] x;
	delete[] x[0];
	delete[] y;
	delete[] is_cate;	
}

void Dataset::init(int n_classes, int n_features) {
	this->n_classes = n_classes;
	this->n_features = n_features;
	x = new ev_pair_t*[this->n_features];
	x[0] = new ev_pair_t[1];

	is_cate = new bool[this->n_features];
	is_init = true;
}

void Dataset::load_data(const std::string& filename, const learn_mode mode) {
	DataReader* dr = new DataReader(filename, n_features, mode);
	std::vector<example_t*> ex_vec;
	m_timer* t = new m_timer();

	if (!is_init) {
		std::cerr << "Please init the Dataset first" << std::endl;
		exit(EXIT_FAILURE);
	}

	t->tic("Loading data from file "+filename+" ...");
	ex_vec = dr->read_examples();
	t->toc("Done.");

	t->tic("Generating dataset ...");
	for (auto it = ex_vec.begin(); it != ex_vec.end(); it++) {
			
	}
	t->toc("Done.");
}

