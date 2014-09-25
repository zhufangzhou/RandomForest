#ifndef __METRICS
#define __METRICS

#include "Utils.h"
#include <iostream>
namespace Metrics {
	double precision(double* y_pred, double* y_true, int size);
	double precision(int* y_pred, double* y_true, int size);
	double precision(double* y_pred, int* y_true, int size);
	double precision(int* y_pred, int* y_true, int size);
	double recall(double* y_pred, double* y_true, int size);
	double recall(int* y_pred, double* y_true, int size);
	double recall(double* y_pred, int* y_true, int size);
	double recall(int* y_pred, int* y_true, int size);
};

#endif