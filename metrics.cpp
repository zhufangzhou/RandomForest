#include "metrics.h"
#include "utils.h"

int* Metrics::gen_label(double* proba, int size, double threshold) {
	int *label = new int[size];
	if (threshold > 1 || threshold < 0) {
		throw "metrics.cpp::gen_label-->\n\tThe `threshold` which is used to generate label must between 0 and 1";
	}
	for (int i = 0; i < size; i++) {
		if (proba[i] > 1 || proba[i] < 0) {
			throw "metrics.cpp::gen_label-->\n\tThe `proba` which is used generate label must between 0 and 1";
		} else if (proba[i] >= threshold) {
			label[i] = 1;
		} else {
			label[i] = 0;
		}
	}
	return label;
}

double Metrics::precision(double *y_pred, double *y_true, int size, double threshold) {
	int *y_pred_i, *y_true_i;
	y_pred_i = Metrics::gen_label(y_pred, size, threshold);
	y_true_i = double2int(y_true, size);
	return Metrics::precision(y_pred_i, y_true_i, size);
}

double Metrics::precision(double *y_pred, int *y_true, int size, double threshold) {
	int *y_pred_i;
	y_pred_i = Metrics::gen_label(y_pred, size, threshold);
	return Metrics::precision(y_pred_i, y_true, size);
}

double Metrics::precision(int *y_pred, double *y_true, int size) {
	int *y_true_i;
	y_true_i = double2int(y_true, size);
	return Metrics::precision(y_pred, y_true_i, size);
}

double Metrics::precision(int *y_pred, int *y_true, int size) {
	int TP, FP, val;
	TP = FP = 0;
	for (int i = 0; i < size; i++) {
		if (y_pred[i] > 1 || y_pred[i] < 0 || y_true[i] > 1 || y_true[i] < 0) {
			std::cerr << "y_pred and y_true must be 0 or 1" << std::endl;
			exit(EXIT_FAILURE);
		}
		val = (y_pred[i] << 1) + y_true[i];
		if (val == 3) TP++;
		if (val == 2) FP++;
	}
	return (double)TP / (TP + FP);
}

double Metrics::recall(double *y_pred, double *y_true, int size, double threshold) {
	int *y_pred_i, *y_true_i;
	y_pred_i = Metrics::gen_label(y_pred, size, threshold);
	y_true_i = double2int(y_true, size);
	return Metrics::recall(y_pred_i, y_true_i, size);
}

double Metrics::recall(double *y_pred, int *y_true, int size, double threshold) {
	int *y_pred_i;
	y_pred_i = Metrics::gen_label(y_pred, size, threshold);
	return Metrics::recall(y_pred_i, y_true, size);
}

double Metrics::recall(int *y_pred, double *y_true, int size) {
	int *y_true_i;
	y_true_i = double2int(y_true, size);
	return Metrics::recall(y_pred, y_true_i, size);
}

double Metrics::recall(int *y_pred, int *y_true, int size) {
	int TP, TN, val;
	TP = TN = 0;
	for (int i = 0; i < size; i++) {
		if (y_pred[i] > 1 || y_pred[i] < 0 || y_true[i] > 1 || y_true[i] < 0) {
			std::cerr << "y_pred and y_true must be 0 or 1" << std::endl;
			exit(EXIT_FAILURE);
		}
		val = (y_pred[i] << 1) + y_true[i];
		if (val == 3) TP++;
		if (val == 1) TN++;
	}
	return (double)TP / (TP + TN);
}
