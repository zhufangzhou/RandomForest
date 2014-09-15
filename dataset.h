#ifndef __DATASET_HEADER
#define __DATASET_HEADER

#include "../utils/Utils.h"
#include <cstdlib>

class Dataset {	
public:
	int sample_size;						// sample size
	int feature_size;						// feature size
	double *X;								// sample
	double *y;								// label
	int *discrete_idx;						// discrete feature idx array
	int discrete_size;						// discreate feature size
	Dataset();
	~Dataset();
	void readText(std::string filename, int feature_size, int *discrete_idx = NULL, int discrete_size = 0);
	void readBinary(std::string filename, int feature_size, int *discrete_idx = NULL, int discrete_size = 0);
	void readBinary(std::string feature_filename, std::string label_filename, int feature_size, int *discrete_idx = NULL, int discrete_size = 0);
};


#endif