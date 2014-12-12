/**
 * @file metrics.h
 * @brief a lot of measures to measure performance
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-12-01
 */
#pragma once

#include <iostream>
#include "utils.h"

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

	double roc_auc_score(double* y_pred, int* y_true, int size);
	double pr_auc_score(double* y_pred, int* y_true, int size);
	double auc(double* x, double* y, int size);
};

