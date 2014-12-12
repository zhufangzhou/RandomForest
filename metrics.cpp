/**
 * @file metrics.h
 * @brief a lot of measures to measure performance
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-12-01
 */
#include "metrics.h"
#include "utils.h"

int* Metrics::gen_label(double* proba, int size, double threshold) {
	int *label = new int[size];
	if (threshold > 1 || threshold < 0) {
		std::cerr << "metrics.cpp::gen_label-->\n\tThe `threshold` which is used to generate label must between 0 and 1" << std::endl;
		exit(EXIT_FAILURE);

	}
	for (int i = 0; i < size; i++) {
		if (proba[i] > 1 || proba[i] < 0) {
			std::cerr << "metrics.cpp::gen_label-->\n\tThe `proba` which is used generate label must between 0 and 1" << std::endl;
			exit(EXIT_FAILURE);
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

double Metrics::roc_auc_score(double* y_pred, int* y_true, int size) {
	int *idx, n_pos = 0, n_neg = 0;
	double ret_auc = 0.0;
	// sort the `y_pred`
	idx = argsort(y_pred, size, DESC);

	for (int i = 0; i < size; i++) {
		if (y_pred[idx[i]] > 1 || y_pred[idx[i]] < 0 || y_true[idx[i]] > 1 || y_true[idx[i]] < 0) {
			std::cerr << "y_true must be 0 or 1 and y_pred must be real number between 0 and 1" << std::endl;
			exit(EXIT_FAILURE);
		}
		
		// count number of positive examples and negtive examples and their ranks.
		if (y_true[idx[i]] == 1) {
			n_pos++;
			ret_auc += size - i;
		} else {
			n_neg++;
		}
	}

	ret_auc = (ret_auc - n_pos*(n_pos+1)/2) / (n_pos+n_neg);
	delete[] idx;
	return ret_auc;
}

double Metrics::pr_auc_score(double* y_pred, int* y_true, int size) {
	int TP = 0, FP = 0, TN = 0, FN = 0;
	int *idx;
	double *x, *y, ret_auc;

	idx = argsort(y_pred, size, DESC);

	// predict all the example to negtive
	for (int i = 0; i < size; i++) {
		if (y_true[idx[i]] == 1) FN++;
		else TN++;
	}

	x = new double[size];
	y = new double[size];
	// add each example to positive group in turn
	for (int i = 0; i < size; i++) {
		if (y_true[idx[i]] == 1) {
			FN--;
			TP++;
		} else {
			TN--;
			FP++;
		}
		x[i] = 1.0 * TP / (TP+FN);
		y[i] = 1.0 * TP / (TP+FP);
	}

	// calculate the area under curve given the points
	ret_auc = auc(x, y, size);

	delete[] x;
	delete[] y;
	delete[] idx;

	return ret_auc;
}

double Metrics::auc(double* x, double* y, int size) {
	int *idx;
	double last_x = 0.0, ret_auc = 0.0;
	
	idx = argsort(x, size, ASC);

	for (int i = 0; i < size; i++) {
		if (x[idx[i]] > 1 || x[idx[i]] < 0 || y[idx[i]] > 1 || y[idx[i]] < 0) {
			std::cerr << "the 2-d cordinate must satisfy 0<=x,y<=1" << std::endl;
			exit(EXIT_FAILURE);
		}
	
		// accumulate the areas of triangles
		ret_auc += (x[idx[i]] - last_x) * y[idx[i]];
		
		last_x = x[idx[i]];
	}
	// free space 
	delete[] idx;
	return ret_auc;
}
