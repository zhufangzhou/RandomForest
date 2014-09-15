#include "dataset.h"

Dataset::Dataset() {
	X = NULL;
	y = NULL;
}

Dataset::~Dataset() {
	if (X != NULL)
		delete[] X;
	if (y != NULL)
		delete[] y;
}


void Dataset::readBinary(std::string filename, int feature_size) {
	FILE* fp = fopen(filename.c_str(), "rb");
	int i = 0;
	timer.tic();
	std::cout << "Start reading dataset from " << filename << std::endl;

	while (fread(y + i, sizeof(double), 1, fp)) {
		if (fread(X + i*feature_size, sizeof(double), feature_size, fp) != feature_size) {
			std::cout << "dataset size is wrong, missing several values" << std::endl;
			exit(EXIT_FAILURE);
		}
		i++;
	}
	
	std::cout << "Finish reading dataset. Sample size: " << i << ", Feature size: " << feature_size << "." << std::endl;
	timer.toc();

	this->sample_size = i;
	this->feature_size = feature_size;
	
	fclose(fp);
}

void Dataset::readBinary(std::string feature_filename, std::string label_filename, int feature_size) {
	FILE *fp_feature = fopen(feature_filename.c_str(), "rb");
	FILE *fp_label = fopen(label_filename.c_str(), "rb");
	int i = 0, j = 0;
	size_t read_count;
	
	timer.tic();
	std::cout << "Start reading dataset from " << filename << std::endl;

	while (read_count = fread(X + i*feature_size, sizeof(double), feature_size, fp_feature)) {
		if(read_count != feature_size) {
			std::cout << "dataset size is wrong, missing several values" << std::endl;
			exit(EXIT_FAILURE);
		}
		i++;
	}
	if (fread(y, sizeof(double), i, fp_label) != i) {
		std::cout << "label file sample size dismatch with feature file." << std::endl;
		exit(EXIT_FAILURE);
	}

	std::cout << "Finish reading dataset. Sample size: " << i << ", Feature size: " << feature_size << "." << std::endl;
	timer.toc();

	this->sample_size = i;
	this->feature_size = feature_size;

	fclose(fp_feature);
	fclose(fp_label);
}

void Dataset::readText(std::string filename, int feature_size) {
	FILE *fp = fopen(filename.c_str(), "r");
	int i;

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
	
	fclose(fp);
}

