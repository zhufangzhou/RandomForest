#ifndef DATASET_HEADER
#define DATASET_HEADER

#include <iostream>
#include <cstdio>
#include <string>
#include "../OpenSource/utils/Utils.h"


class Dataset {
private:
	int sample_size;						// sample size
	int feature_size;						// feature size
	double *X;					// sample
	double *y;					// label
public:
	Dataset();
	~Dataset();
	void readText(std::string filename, int feature_size);
	void readBinary(std::string filename, int feature_size);
	void readBinary(std::string feature_filename, std::string label_filename, int feature_size);
};


#endif