#include "metrics.h"
#include "utils.h"
double Metrics::precision(double *y_pred, double *y_true, int size) {
	int *y_pred_i, *y_true_i;
	y_pred_i = double2int(y_pred, size);
	y_true_i = double2int(y_true, size);
	return Metrics::precision(y_pred_i, y_true_i, size);
}

double Metrics::precision(int *y_pred, double *y_true, int size) {
	int *y_true_i;
	y_true_i = double2int(y_true, size);
	return Metrics::precision(y_pred, y_true_i, size);
}

double Metrics::precision(double *y_pred, int *y_true, int size) {
	int *y_pred_i;
	y_pred_i = double2int(y_pred, size);
	return Metrics::precision(y_pred_i, y_true, size);
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

double Metrics::recall(double *y_pred, double *y_true, int size) {
	int *y_pred_i, *y_true_i;
	y_pred_i = double2int(y_pred, size);
	y_true_i = double2int(y_true, size);
	return Metrics::recall(y_pred_i, y_true_i, size);
}

double Metrics::recall(int *y_pred, double *y_true, int size) {
	int *y_true_i;
	y_true_i = double2int(y_true, size);
	return Metrics::recall(y_pred, y_true_i, size);
}

double Metrics::recall(double *y_pred, int *y_true, int size) {
	int *y_pred_i;
	y_pred_i = double2int(y_pred, size);
	return Metrics::recall(y_pred_i, y_true, size);
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
