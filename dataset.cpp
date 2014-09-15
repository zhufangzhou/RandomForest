#include "dataset.h"

Dataset::Dataset() {
	// initialize X and y, otherwise will error when call realloc
	X = NULL;
	y = NULL;
	// discrete_idx = NULL;
	// discrete_size = 0;
	discrete_mask = NULL;	
}

Dataset::~Dataset() {
	delete[] X;
	delete[] y;
	delete[] discrete_mask;
}


void Dataset::readBinary(std::string filename, int feature_size, int *discrete_idx, int discrete_size) {
	FILE* fp = fopen(filename.c_str(), "rb");
	int i = 0;
	double *X_buf, y_buf;
	X_buf = new double[feature_size];

	timer.tic();
	std::cout << "Start reading dataset from " << filename << std::endl;

	// first read a sample into buffer
	while (fread(&y_buf, sizeof(double), 1, fp)) {	
		if (fread(X_buf, sizeof(double), feature_size, fp) != feature_size) {
			std::cout << "dataset size is wrong, missing several values" << std::endl;
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
	
	std::cout << "Finish reading dataset. Sample size: " << i << ", Feature size: " << feature_size << "." << std::endl;
	timer.toc();

	// set sample size and feature size
	this->sample_size = i;
	this->feature_size = feature_size;
	
	// determine which features is discrete
	discrete_mask = new bool[feature_size];
	memset(discrete_mask, false, sizeof(bool)*feature_size);
	if (!(discrete_idx == 0 || discrete_idx == NULL)) {
		// this->discrete_idx = discrete_idx;
		// this->discrete_size = discrete_size;
		for (int j = 0; j < discrete_size; j++) {
			discrete_mask[discrete_idx[j]] = true;
		}
	}

	// delete buffer
	delete[] X_buf;
	fclose(fp);
}

void Dataset::readBinary(std::string feature_filename, std::string label_filename, int feature_size, int *discrete_idx, int discreate_size) {
	FILE *fp_feature = fopen(feature_filename.c_str(), "rb");
	FILE *fp_label = fopen(label_filename.c_str(), "rb");
	int i = 0, j = 0;
	double *X_buf;
	size_t read_count;
	
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

	// determine which features is discrete
	discrete_mask = new bool[feature_size];
	memset(discrete_mask, false, sizeof(bool)*feature_size);
	if (!(discrete_idx == 0 || discrete_idx == NULL)) {
		// this->discrete_idx = discrete_idx;
		// this->discrete_size = discrete_size;
		for (int j = 0; j < discreate_size; j++) {
			discrete_mask[discrete_idx[j]] = true;
		}
	}

	delete[] X_buf;
	fclose(fp_feature);
	fclose(fp_label);
}

void Dataset::readText(std::string filename, int feature_size, int *discrete_idx, int discreate_size) {
	FILE *fp = fopen(filename.c_str(), "r");
	const int MAX_LINE = feature_size * 20;
	const char* DELIMITER = " ";
	double *X_buf = new double[feature_size], y_buf;
	char *line = new char[MAX_LINE], *pch;
	int i;

	timer.tic();
	std::cout << "Start reading dataset from " << filename << std::endl;
	
	// read a line and split to get label and each feature value	
	while (fgets(line, MAX_LINE, fp)) {
		pch = strtok(line, DELIMITER);			// read label
		if (pch == NULL || pch == "") {
			std::cout << "Line " << i+1 << " miss label." << std::endl;
			exit(EXIT_FAILURE);
		}
		y_buf = atof(pch);

		// read features
		for (int j = 0; j < feature_size - 1; j++) {
			pch = strtok(NULL, DELIMITER);
			if (pch == NULL || pch == "") {
				std::cout << "Line " << i+1 << " miss feature" << "." << std::endl;
				exit(EXIT_FAILURE);
			}
			X_buf[j] = atof(pch);
		}
		// read last feature
		pch = strtok(NULL, "\n");
		if (pch == NULL || pch == "") {
			std::cout << "Line " << i+1 << " miss feature" << "." << std::endl;
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

	// determine which features is discrete
	discrete_mask = new bool[feature_size];
	memset(discrete_mask, false, sizeof(bool)*feature_size);
	if (!(discrete_idx == 0 || discrete_idx == NULL)) {
		// this->discrete_idx = discrete_idx;
		// this->discrete_size = discrete_size;
		for (int j = 0; j < discreate_size; j++) {
			discrete_mask[discrete_idx[j]] = true;
		}
	}
	
	delete[] line;
	delete[] X_buf;
	fclose(fp);
}

