#ifndef __DATASET_HEADER
#define __DATASET_HEADER

//#include <iostream>
//#include <cstdio>
//#include <string>
//#include <vector>
#include "../utils/Utils.h"


class Dataset {
private:
	int sample_size;						// sample size
	int feature_size;						// feature size
//	std::vector<double> X;					// sample
//	std::vector<double> y;					// label
	double *X;
	double *y;
public:
	Dataset();
	~Dataset();
	void readText(std::string filename, int feature_size);
	void readBinary(std::string filename, int feature_size);
	void readBinary(std::string feature_filename, std::string label_filename, int feature_size);
};


#endif