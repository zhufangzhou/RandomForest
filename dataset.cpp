#include "dataset.h"

Dataset::Dataset() {
	// initialize X and y, otherwise will error when call realloc
	X = NULL;
	y = NULL;
	discrete_idx = NULL;
	discrete_size = 0;
	discrete_mask = NULL;	
}

Dataset::~Dataset() {
	delete[] X;
	delete[] y;
	delete[] discrete_mask;
}

void Dataset::reset() {
	delete[] X;
	delete[] y;
	sample_size = 0;
	feature_size = 0;
}

void Dataset::set_dataset(double *X, double *y, int sample_size, int feature_size, int *discrete_idx, int discrete_size) {
	set_dataset(X, sample_size, feature_size, discrete_idx, discrete_size);
	if (y != NULL) {
		this->y = new double[sample_size];
		memcpy(this->y, y, sizeof(double)*sample_size);
	}
}

void Dataset::set_dataset(double *X, int sample_size, int feature_size, int *discrete_idx, int discrete_size) {
	// first reset the dataset
	reset();
	// initialize dataset
	this->sample_size = sample_size;
	this->feature_size = feature_size;
	
	if (X != NULL) {
		this->X = new double[sample_size*feature_size];
		memcpy(this->X, X, sizeof(double)*sample_size*feature_size);
	}

	// deal with discrete feature
	if (discrete_idx != 0 && discrete_idx != NULL)
		handle_discrete_feature(discrete_idx, discrete_size);
}

void Dataset::handle_discrete_feature(int *discrete_idx, int discrete_size) {
	// use set to obatain unique discrete values
	std::unordered_set<discrete_t> ss;
	int counts = 0;

	
	
	if (!(discrete_idx == 0 || discrete_idx == NULL)) {
		// determine which features is discrete
		discrete_mask = new int[feature_size];
		memset(discrete_mask, 0xff, sizeof(int)*feature_size);
		
		this->discrete_idx = discrete_idx;
		this->discrete_size = discrete_size;
		
		this->discrete_value = new discrete_t*[discrete_size];
		for (int j = 0; j < discrete_size; j++) {
			// update the discrete mask, if -1 is continuous then the index of discrete_value
			discrete_mask[discrete_idx[j]] = j;
			// clear the unordered_set for a new discrete feature
			ss.clear();
			// insert all the feature value of this discrete feature into unordered_set and get distinct value
			for (int i = 0; i < sample_size; i++) {
				ss.insert((discrete_t) X[i*feature_size + discrete_idx[j]]);
			}
			discrete_value[j] = new discrete_t[ss.size() + 1];
			// the first element for each discrete feature is its distinct value count --> discrete_value[j][0]
			discrete_value[j][0] = 0;
			// iterate the unordered_set to get all the distinct value for this discrete feature
			for (auto it = ss.begin(); it != ss.end(); it++) {
				discrete_value[j][(int)++discrete_value[j][0]] = *it;
			}
		}
	}
	delete[] discrete_mask;
}

void Dataset::readBinary(std::string filename, int feature_size, bool is_train, int *discrete_idx, int discrete_size) {
	FILE* fp = fopen(filename.c_str(), "rb");
	int i = 0;
	double *X_buf, y_buf;
	size_t read_count;
	
	// first reset the dataset
	reset();
	
	X_buf = new double[feature_size];

	timer.tic();
	std::cout << "Start reading dataset from " << filename << std::endl;

	if(is_train) {
		// first read a sample into buffer
		while (fread(&y_buf, sizeof(double), 1, fp)) {	
			if (fread(X_buf, sizeof(double), feature_size, fp) != feature_size) {
				std::cerr << "dataset size is wrong, missing several values" << std::endl;
				exit(EXIT_FAILURE);
			}
			
			// sample size add one
			i++;
			// expand space to store current sample
			X = (double*) realloc(X, sizeof(double)*i*feature_size);
			y = (double*) realloc(y, sizeof(double)*i);
			// copy buffer to dataset
			memcpy(X + (i-1)*feature_size, X_buf, sizeof(double)*feature_size);
			memcpy(y + (i-1), &y_buf, sizeof(double));
		}
	} else {
		while (read_count = fread(X_buf, sizeof(double), feature_size, fp)) {
			if(read_count != feature_size) {
				std::cout << "dataset size is wrong, missing several values" << std::endl;
				exit(EXIT_FAILURE);
			}
			i++;
			X = (double*) realloc(X, sizeof(double)*i*feature_size);
			memcpy(X + (i-1)*feature_size, X_buf, sizeof(double)*feature_size);
		}
	}
	
	std::cout << "Finish reading dataset. Sample size: " << i << ", Feature size: " << feature_size << "." << std::endl;
	timer.toc();

	// set sample size and feature size
	this->sample_size = i;
	this->feature_size = feature_size;
	
	// deal with discrete feature
	handle_discrete_feature(discrete_idx, discrete_size);

	// delete buffer
	delete[] X_buf;
	fclose(fp);
}

void Dataset::readBinary(std::string feature_filename, std::string label_filename, int feature_size,
						 int *discrete_idx, int discreate_size) {
	FILE *fp_feature = fopen(feature_filename.c_str(), "rb");
	FILE *fp_label = fopen(label_filename.c_str(), "rb");
	int i = 0, j = 0;
	double *X_buf;
	size_t read_count;
	
	// first reset the dataset
	reset();

	X_buf = (double*) malloc(sizeof(double)*feature_size);

	timer.tic();
	std::cout << "Start reading dataset from " << feature_filename << " and " << label_filename << std::endl;

	while (read_count = fread(X_buf, sizeof(double), feature_size, fp_feature)) {
		if(read_count != feature_size) {
			std::cout << "dataset size is wrong, missing several values" << std::endl;
			exit(EXIT_FAILURE);
		}
		i++;
		X = (double*) realloc(X, sizeof(double)*i*feature_size);
		memcpy(X + (i-1)*feature_size, X_buf, sizeof(double)*feature_size);
	}

	y = (double*) malloc(sizeof(double)*i);
	if (fread(y, sizeof(double), i, fp_label) != i) {
		std::cout << "label file sample size dismatch with feature file." << std::endl;
		exit(EXIT_FAILURE);
	}

	std::cout << "Finish reading dataset. Sample size: " << i << ", Feature size: " << feature_size << "." << std::endl;
	timer.toc();

	this->sample_size = i;
	this->feature_size = feature_size;

	// deal with discrete feature
	handle_discrete_feature(discrete_idx, discrete_size);

	delete[] X_buf;
	fclose(fp_feature);
	fclose(fp_label);
}

void Dataset::readText(std::string filename, int feature_size, bool is_train, int *discrete_idx, int discreate_size) {
	FILE *fp = fopen(filename.c_str(), "r");
	const int MAX_LINE = feature_size * 20;
	const char* DELIMITER = " ";
	double *X_buf = new double[feature_size], y_buf;
	char *line = new char[MAX_LINE], *pch;
	int i = 0, start;

	// first reset the dataset
	reset();

	std::cout << "Start reading dataset from " << filename << std::endl;
	timer.tic();
	
	// read a line and split to get label and each feature value	
	while (fgets(line, MAX_LINE, fp)) {
		pch = strtok(line, DELIMITER);			// read label
		if (pch == NULL || pch == "") {
			std::cerr << "Line " << i+1 << " miss label." << std::endl;
			exit(EXIT_FAILURE);
		}
		if (is_train) {
			y_buf = atof(pch);
			start = 0;
		} else {								// predict do not need to read label
			X_buf[0] = atof(pch);
			start = 1;
		}

		// read features
		for (int j = start; j < feature_size - 1; j++) {
			pch = strtok(NULL, DELIMITER);
			if (pch == NULL || pch == "") {
				std::cerr << "Line " << i+1 << " miss feature" << "." << std::endl;
				exit(EXIT_FAILURE);
			}
			X_buf[j] = atof(pch);
		}
		// read last feature
		pch = strtok(NULL, "\n");
		if (pch == NULL || pch == "") {
			std::cerr << "Line " << i+1 << " miss feature" << "." << std::endl;
			exit(EXIT_FAILURE);
		}
		X_buf[feature_size-1] = atof(pch);

		// sample size add one
		i++;
		// expand space to store current sample
		X = (double*) realloc(X, sizeof(double)*i*feature_size);
		y = (double*) realloc(y, sizeof(double)*i);
		// copy buffer to dataset
		memcpy(X + (i-1)*feature_size, X_buf, sizeof(double)*feature_size);
		memcpy(y + (i-1), &y_buf, sizeof(double));
	}

	std::cout << "Finish reading dataset. Sample size: " << i << ", Feature size: " << feature_size << "." << std::endl;
	
	timer.toc();
	
	this->sample_size = i;
	this->feature_size = feature_size;

	// deal with discrete feature
	handle_discrete_feature(discrete_idx, discrete_size);
	
	delete[] line;
	delete[] X_buf;
	fclose(fp);
}

