/**
 * @file metrics.h
 * @brief a lot of measures to measure performance
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-12-01
 */
#pragma once

#include <iostream>
#include <fstream>
#include <algorithm>
#include "utils.h"

namespace Metrics {
	int* gen_label(float* proba, int size, float threshold);

	/* precision for binary class */
	float precision(float* y_pred, float* y_true, int size, float threshold = 0.5);
	float precision(float* y_pred, int* y_true, int size, float threshold = 0.5);
	float precision(int* y_pred, float* y_true, int size);
	float precision(int* y_pred, int* y_true, int size);

	/* precision for multi-class */
	float precision_multi(float* y_pred, float* y_true, int n_classes, int size);
	float precision_multi(float* y_pred, int* y_true, int n_classes, int size);
	float precision_multi(int* y_pred, float* y_true, int n_classes, int size);
	float precision_multi(int* y_pred, int* y_true, int n_classes, int size);

	/* recall for binary class */
	float recall(float* y_pred, float* y_true, int size, float threshold = 0.5);
	float recall(float* y_pred, int* y_true, int size, float threshold = 0.5);
	float recall(int* y_pred, float* y_true, int size);
	float recall(int* y_pred, int* y_true, int size);

	/* recall for multi-class */
	float recall_multi(float* y_pred, float* y_true, int n_classes, int size);
	float recall_multi(float* y_pred, int* y_true, int n_classes, int size);
	float recall_multi(int* y_pred, float* y_true, int n_classes, int size);
	float recall_multi(int* y_pred, int* y_true, int n_classes, int size);

	/* f1-score for binary class */
	float f1_score(float* y_pred, float* y_true, int size, float threshold = 0.5);
	float f1_score(float* y_pred, int* y_true, int size, float threshold = 0.5);
	float f1_score(int* y_pred, float* y_true, int size);
	float f1_score(int* y_pred, int* y_true, int size);

	/* f1-score for multi-class*/
	float f1_score_multi(float* y_pred, float* y_true, int n_classes, int size);
	float f1_score_multi(int* y_pred, float* y_true, int n_classes, int size);
	float f1_score_multi(float* y_pred, int* y_true, int n_classes, int size);
	float f1_score_multi(int* y_pred, int* y_true, int n_classes, int size);

	float roc_auc_score(float* y_pred, int* y_true, int size);
	float roc_auc_score_multi(float* y_pred, int* y_true, int n_classes, int size);
	float pr_auc_score(float* y_pred, int* y_true, int size);
	float pr_auc_score_multi(float* y_pred, int* y_true, int n_classes, int size);
	float auc(float* x, float* y, int size);

	void performance_report(float* y_pred, int* y_true, int size, float threshold = 0.5);
	void performance_report(const std::string& filename, float* y_pred, int* y_true, int size, float threshold = 0.5);
};

