#include "dataset.h"

Dataset::Dataset() {
	// initialize X and y, otherwise will error when call realloc
	X = NULL;
	y = NULL;
}

Dataset::~Dataset() {
	delete[] X;
	delete[] y;
}


void Dataset::readBinary(std::string filename, int feature_size) {
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
	
	// delete buffer
	delete[] X_buf;
	fclose(fp);
}

void Dataset::readBinary(std::string feature_filename, std::string label_filename, int feature_size) {
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

	delete[] X_buf;
	fclose(fp_feature);
	fclose(fp_label);
}

void Dataset::readText(std::string filename, int feature_size) {
	/*FILE *fp = fopen(filename.c_str(), "r");
	int i;
	const 
	timer.tic();
	std::cout << "Start reading dataset from " << filename << std::endl;
	
	while(~fscanf(fp, "%d", &y[i])) {
		for (int j = 0; j < feature_size; j++) {
			fscanf(fp, "%lf", &X[i*feature_size + j]);
		}
		i++;
	}

	std::cout << "Finish reading dataset. Sample size: " << i << ", Feature size: " << feature_size << "." << std::endl;
	timer.toc();
	
	this->sample_size = i;
	this->feature_size = feature_size;
	
	fclose(fp);*/
}

