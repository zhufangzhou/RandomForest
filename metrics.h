#ifndef __METRICS
#define __METRICS

#include <iostream>

namespace Metrics {
	int* gen_label(double* proba, int size, double threshold);

	double precision(double* y_pred, double* y_true, int size, double threshold = 0.5);
	double precision(double* y_pred, int* y_true, int size, double threshold = 0.5);
	double precision(int* y_pred, double* y_true, int size);
	double precision(int* y_pred, int* y_true, int size);

	double recall(double* y_pred, double* y_true, int size, double threshold = 0.5);
	double recall(double* y_pred, int* y_true, int size, double threshold = 0.5);
	double recall(int* y_pred, double* y_true, int size);
	double recall(int* y_pred, int* y_true, int size);

	double f1_score(double* y_pred, double *y_true, int size, double threshold = 0.5);
	double f1_score(double* y_pred, int* y_true, int size, double threshold = 0.5);
	double f1_score(int* y_pred, double* y_true, int size);
	double f1_score(int* y_pred, int* y_true, int size);
};

#endif
