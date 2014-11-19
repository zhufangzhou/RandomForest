#include "dataset.h"

example_t::example_t() {
	nnz = 0;
	y = -1;
	fea_id = new int[1];
	fea_value = new feature_t[1];
}

example_t::~example_t() {
	delete[] fea_id;
	felete[] fea_value;
}

void example_t::push_back(int id, feature_t value) {
	fea_id = (int*)realloc(fea_id, nnz+1);
	fea_value = (feature_t*)realloc(fea_value, nnz+1);

	fea_id[nnz] = id;
	fea_value[nnz] = value;

	nnz++;
}

DataReader::DataReader(std::string filename, int n_features) {
	ifs.open(filename.c_str(), ios::binary);
	if (ifs == NULL) {
		std::cerr << "Can not open file " << filename << " ." << std::endl;
		exit(EXIT_FAILURE);
	}
	this->n_features = n_features;
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
	
	ifs >> ret->y;
	getline(ifs, line);
	p_pos = 0;
	c_pos = line.find(' ', 0);
	
	while (p_pos <= c_pos) {
		p_pos = c_pos + 1;
		c_pos = line.find(':', p_pos);
		t_str = line.substr(p_pos, c_pos - p_pos);
		feature_id = atoi(t_str.c_str()) - startIndex;

		p_pos = c_pos + 1;
		c_pos = line.find(' ', p_pos);
		feature_value = atof(line.substr(p_pos, c_pos - p_pos));

		ret->push_back(feature_id, feature_value);
	}
	return ret;	
}

ev_pair_t* DataReader::read_examples() {
	example_t* single = new example_t();
	ev_pair_t
	while( (single=read_an_example()) != NULL ) {

	}
}
