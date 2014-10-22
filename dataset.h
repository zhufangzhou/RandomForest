#ifndef __DATASET_HEADER
#define __DATASET_HEADER

#include <cstdlib>
#include <unordered_set>

#define discrete_t double
#define TRAIN true
#define PREDICT false

class Dataset {	
private:
	void handle_discrete_feature(int *discrete_idx, int discrete_size);
	void reset();
public:
	int sample_size;						// sample size
	int feature_size;						// feature size
	double *X;								// sample
	double *y;								// label

	// discrete feature
	int *discrete_mask;						// not -1 --> discrete feature, -1 --> continuous feature
	discrete_t **discrete_value;			// discrete feature value list
	int *discrete_idx;						// discrete feature idx array
	int discrete_size;						// discreate feature size

	// constructor
	Dataset();
	~Dataset();
	
	void set_dataset(double *X, double *y, int sample_size, int feature_size, bool is_copy, int *discrete_idx = NULL, int discrete_size = 0);
	void set_dataset(double *X, int sample_size, int feature_size, bool is_copy, int *discrete_idx = NULL, int discrete_size = 0);
	void set_dataset(Dataset ds, bool is_copy);

	void readText(std::string filename, int feature_size, bool is_train, int *discrete_idx = NULL, int discrete_size = 0);
	
	void readBinary(std::string filename, int feature_size, bool is_train, int *discrete_idx = NULL, int discrete_size = 0);
	void readBinary(std::string feature_filename, std::string label_filename, int feature_size, 
					int *discrete_idx = NULL, int discrete_size = 0);
};


#endif
