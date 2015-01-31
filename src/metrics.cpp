/**
 * @file metrics.h
 * @brief a lot of measures to measure performance
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-12-01
 */
#include "metrics.h"
#include "utils.h"

int* Metrics::gen_label(float* proba, int size, float threshold) {
	int *label = new int[size];
	if (threshold > 1 || threshold < 0) {
		std::cerr << "metrics.cpp::gen_label-->\n\tThe `threshold` which is used to generate label must between 0 and 1" << std::endl;
		exit(EXIT_FAILURE);

	}
	for (int i = 0; i < size; i++) {
		if (proba[i] > 1 || proba[i] < 0) {
			std::cerr << "metrics.cpp::gen_label-->\n\tThe `proba` which is used generate label must between 0 and 1. Here is #" << i << ": " << proba[i] << std::endl;
			exit(EXIT_FAILURE);
		} else if (proba[i] >= threshold) {
			label[i] = 1;
		} else {
			label[i] = 0;
		}
	}
	return label;
}

float Metrics::precision(float *y_pred, float *y_true, int size, float threshold) {
	int *y_pred_i = nullptr, *y_true_i = nullptr;
	float ret_precision;
	y_pred_i = Metrics::gen_label(y_pred, size, threshold);
	y_true_i = float2int(y_true, size);

	ret_precision = Metrics::precision(y_pred_i, y_true_i, size);

	if (y_pred_i != nullptr) {
		delete[] y_pred_i;
		y_pred_i = nullptr;
	}
	if (y_true_i != nullptr) {
		delete[] y_true_i;
		y_true_i = nullptr;
	}

	return ret_precision;
}

float Metrics::precision(float *y_pred, int *y_true, int size, float threshold) {
	int *y_pred_i = nullptr;
	float ret_precision;
	y_pred_i = Metrics::gen_label(y_pred, size, threshold);

	ret_precision = Metrics::precision(y_pred_i, y_true, size);

	if (y_pred_i != nullptr) {
		delete[] y_pred_i;
		y_pred_i = nullptr;
	}
	return ret_precision;
}

float Metrics::precision(int *y_pred, float *y_true, int size) {
	int *y_true_i = nullptr;
	float ret_precision;
	y_true_i = float2int(y_true, size);

	ret_precision = Metrics::precision(y_pred, y_true_i, size);

	if (y_true_i != nullptr) {
		delete[] y_true_i;
		y_true_i = nullptr;
	}

	return ret_precision;
}

float Metrics::precision(int *y_pred, int *y_true, int size) {
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
	if (TP+FP == 0) return 0;
	else return (float)TP / (TP + FP);
}

float Metrics::precision_multi(float* y_pred, float* y_true, int n_classes, int size) {
	int *y_pred_i = nullptr, *y_true_i = nullptr, idx;
	float max_proba, avg_precision;

	/* find the class with maximum probability for each example */
	y_pred_i = new int[size];
	for (int i = 0; i < size; i++) {
		max_proba = 0.0;
		for (int c = 0; c < n_classes; c++) {
			idx = i + c*size;
			if (y_pred[idx] > max_proba) {
				max_proba = y_pred[idx];
				y_pred_i[i] = c;
			}
		}
	}

	y_true_i = float2int(y_true, size);
	avg_precision = Metrics::precision_multi(y_pred_i, y_true_i, n_classes, size);

	/* free space */
	if (y_pred_i != nullptr) {
		delete[] y_pred_i;
		y_pred_i = nullptr;
	}
	if (y_true_i != nullptr) {
		delete[] y_true_i;
		y_true_i = nullptr;
	}

	return avg_precision;
}

float Metrics::precision_multi(float* y_pred, int* y_true, int n_classes, int size) {
	int *y_pred_i = nullptr, idx;
	float max_proba, avg_precision;

	/* find the class with maximum probability for each example */
	y_pred_i = new int[size];
	for (int i = 0; i < size; i++) {
		max_proba = 0.0;
		for (int c = 0; c < n_classes; c++) {
			idx = i + c*size;
			if (y_pred[idx] > max_proba) {
				max_proba = y_pred[idx];
				y_pred_i[i] = c;
			}
		}
	}

	avg_precision = Metrics::precision_multi(y_pred_i, y_true, n_classes, size);

	/* free space */
	if (y_pred_i != nullptr) {
		delete[] y_pred_i;
		y_pred_i = nullptr;
	}

	return avg_precision;
}

float Metrics::precision_multi(int* y_pred, float* y_true, int n_classes, int size) {
	int *y_true_i = nullptr;
	float avg_precision;
	y_true_i = float2int(y_true, size);

	avg_precision = Metrics::precision_multi(y_pred, y_true_i, n_classes, size);

	if (y_true_i != nullptr) {
		delete[] y_true_i;
		y_true_i = nullptr;
	}

	return avg_precision;
}

float Metrics::precision_multi(int* y_pred, int* y_true, int n_classes, int size) {
	int *total_predicted = nullptr, *true_positive = nullptr;
	/* check for `y_true` to determine all the elements are between 0 and n_classes */
	for (int i = 0; i < size; i++) {
		if (y_true[i] < 0 || y_true[i] >= n_classes) {
			std::cerr << "ERROR:Metrics::precision_multi: #" << i << " element in `y_true` is " << y_true[i] << " which is not between 0 and n_classes." << std::endl;
			exit(EXIT_FAILURE);
		}
		if (y_pred[i] < 0 || y_pred[i] >= n_classes) {
			std::cerr << "ERROR:Metrics::precision_multi: #" << i << " element in `y_pred` is " << y_pred[i] << " which is not between 0 and n_classes." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	/* allocate space and initialize to zero */
	true_positive = new int[n_classes]();
	total_predicted = new int[n_classes]();

	/* count true positive and total predicted for each class */
	for (int i = 0; i < size; i++) {
		if (y_pred[i] == y_true[i]) {
			true_positive[y_pred[i]]++;
		}
		total_predicted[y_pred[i]]++;
	}

	float avg_precision = 0.0;
	for (int c = 0; c < n_classes; c++) {
		avg_precision += 1.0 * true_positive[c] / total_predicted[c];
	}
	avg_precision /= n_classes;

	/* free space */
	if (total_predicted != nullptr) {
		delete[] total_predicted;
		total_predicted = nullptr;
	}
	if (true_positive != nullptr) {
		delete[] true_positive;
		true_positive = nullptr;
	}

	return avg_precision;
}

float Metrics::recall(float *y_pred, float *y_true, int size, float threshold) {
	int *y_pred_i, *y_true_i;
	y_pred_i = Metrics::gen_label(y_pred, size, threshold);
	y_true_i = float2int(y_true, size);
	return Metrics::recall(y_pred_i, y_true_i, size);
}

float Metrics::recall(float *y_pred, int *y_true, int size, float threshold) {
	int *y_pred_i;
	y_pred_i = Metrics::gen_label(y_pred, size, threshold);
	return Metrics::recall(y_pred_i, y_true, size);
}

float Metrics::recall(int *y_pred, float *y_true, int size) {
	int *y_true_i;
	y_true_i = float2int(y_true, size);
	return Metrics::recall(y_pred, y_true_i, size);
}

float Metrics::recall(int *y_pred, int *y_true, int size) {
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
	if (TP+TN == 0) return 0;
	else return (float)TP / (TP + TN);
}

float Metrics::recall_multi(float* y_pred, float* y_true, int n_classes, int size) {
	int *y_pred_i = nullptr, *y_true_i = nullptr, idx;
	float max_proba, avg_recall;

	/* find the class with maximum probability for each example */
	y_pred_i = new int[size];
	for (int i = 0; i < size; i++) {
		max_proba = 0.0;
		for (int c = 0; c < n_classes; c++) {
			idx = i + c*size;
			if (y_pred[idx] > max_proba) {
				max_proba = y_pred[idx];
				y_pred_i[i] = c;
			}
		}
	}

	y_true_i = float2int(y_true, size);
	avg_recall= Metrics::recall_multi(y_pred_i, y_true_i, n_classes, size);

	/* free space */
	if (y_pred_i != nullptr) {
		delete[] y_pred_i;
		y_pred_i = nullptr;
	}
	if (y_true_i != nullptr) {
		delete[] y_true_i;
		y_true_i = nullptr;
	}

	return avg_recall;
}

float Metrics::recall_multi(float* y_pred, int* y_true, int n_classes, int size) {
	int *y_pred_i = nullptr, idx;
	float max_proba, avg_recall;

	/* find the class with maximum probability for each example */
	y_pred_i = new int[size];
	for (int i = 0; i < size; i++) {
		max_proba = 0.0;
		for (int c = 0; c < n_classes; c++) {
			idx = i + c*size;
			if (y_pred[idx] > max_proba) {
				max_proba = y_pred[idx];
				y_pred_i[i] = c;
			}
		}
	}

	avg_recall = Metrics::recall_multi(y_pred_i, y_true, n_classes, size);

	/* free space */
	if (y_pred_i != nullptr) {
		delete[] y_pred_i;
		y_pred_i = nullptr;
	}

	return avg_recall;
}

float Metrics::recall_multi(int* y_pred, float* y_true, int n_classes, int size) {
	int *y_true_i = nullptr;
	float avg_recall;
	y_true_i = float2int(y_true, size);

	avg_recall = Metrics::recall_multi(y_pred, y_true_i, n_classes, size);

	if (y_true_i != nullptr) {
		delete[] y_true_i;
		y_true_i = nullptr;
	}

	return avg_recall;
}

float Metrics::recall_multi(int* y_pred, int* y_true, int n_classes, int size) {
	int *total_label = nullptr, *true_positive = nullptr;
	/* check for `y_true` to determine all the elements are between 0 and n_classes */
	for (int i = 0; i < size; i++) {
		if (y_true[i] < 0 || y_true[i] >= n_classes) {
			std::cerr << "ERROR:Metrics::precision_multi: #" << i << " element in `y_true` is " << y_true[i] << " which is not between 0 and n_classes." << std::endl;
			exit(EXIT_FAILURE);
		}
		if (y_pred[i] < 0 || y_pred[i] >= n_classes) {
			std::cerr << "ERROR:Metrics::precision_multi: #" << i << " element in `y_pred` is " << y_pred[i] << " which is not between 0 and n_classes." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	/* allocate space and initialize to zero */
	true_positive = new int[n_classes]();
	total_label = new int[n_classes]();

	/* count true positive and total label for each class */
	for (int i = 0; i < size; i++) {
		if (y_pred[i] == y_true[i]) {
			true_positive[y_pred[i]]++;
		}
		total_label[y_true[i]]++;
	}

	float avg_recall = 0.0;
	for (int c = 0; c < n_classes; c++) {
		avg_recall += 1.0 * true_positive[c] / total_label[c];
	}
	avg_recall /= n_classes;

	/* free space */
	if (total_label!= nullptr) {
		delete[] total_label;
		total_label= nullptr;
	}
	if (true_positive != nullptr) {
		delete[] true_positive;
		true_positive = nullptr;
	}

	return avg_recall;
	
}

float Metrics::f1_score(float* y_pred, float *y_true, int size, float threshold) {
	float precision, recall;	
	precision = Metrics::precision(y_pred, y_true, size, threshold);
	recall = Metrics::recall(y_pred, y_true, size, threshold);

	return 2*precision*recall/(precision+recall);
}

float Metrics::f1_score(float* y_pred, int* y_true, int size, float threshold) {
	float precision, recall;	
	precision = Metrics::precision(y_pred, y_true, size, threshold);
	recall = Metrics::recall(y_pred, y_true, size, threshold);

	return 2*precision*recall/(precision+recall);
}

float Metrics::f1_score(int* y_pred, float* y_true, int size) {
	float precision, recall;	
	precision = Metrics::precision(y_pred, y_true, size);
	recall = Metrics::recall(y_pred, y_true, size);

	return 2*precision*recall/(precision+recall);
}

float Metrics::f1_score(int* y_pred, int* y_true, int size) {
	float precision, recall;	
	precision = Metrics::precision(y_pred, y_true, size);
	recall = Metrics::recall(y_pred, y_true, size);

	return 2*precision*recall/(precision+recall);
}

float Metrics::f1_score_multi(int* y_pred, int* y_true, int n_classes, int size) {
	float precision, recall;
	precision = Metrics::precision_multi(y_pred, y_true, n_classes, size);
	recall = Metrics::recall_multi(y_pred, y_true, n_classes, size);

	return 2*precision*recall/(precision+recall);
}

float Metrics::f1_score_multi(float* y_pred, int* y_true, int n_classes, int size) {
	float precision, recall;
	precision = Metrics::precision_multi(y_pred, y_true, n_classes, size);
	recall = Metrics::recall_multi(y_pred, y_true, n_classes, size);

	return 2*precision*recall/(precision+recall);
}

float Metrics::f1_score_multi(int* y_pred, float* y_true, int n_classes, int size) {
	float precision, recall;
	precision = Metrics::precision_multi(y_pred, y_true, n_classes, size);
	recall = Metrics::recall_multi(y_pred, y_true, n_classes, size);

	return 2*precision*recall/(precision+recall);
}

float Metrics::f1_score_multi(float* y_pred, float* y_true, int n_classes, int size) {
	float precision, recall;
	precision = Metrics::precision_multi(y_pred, y_true, n_classes, size);
	recall = Metrics::recall_multi(y_pred, y_true, n_classes, size);

	return 2*precision*recall/(precision+recall);
}

float Metrics::roc_auc_score(float* y_pred, int* y_true, int size) {
	int *idx, n_pos = 0, n_neg = 0;
	float ret_auc = 0.0;
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

	ret_auc = (ret_auc - n_pos*(n_pos+1)/2) / (n_pos*n_neg);
	delete[] idx;
	return ret_auc > 1.0 ? 1.0 : ret_auc;
}

float Metrics::roc_auc_score_multi(float* y_pred, int* y_true, int n_classes, int size) {
	float avg_roc_auc_score = 0.0;	
	int *y_true_oneVSrest = new int[size];
	float *y_pred_oneVSrest;

	for (int c = 0; c < n_classes; c++) {
		y_pred_oneVSrest = y_pred + c*size;	
		for (int i = 0; i < size; i++) {
			y_true_oneVSrest[i] = y_true[i] == c ? 1 : 0;
		}
		avg_roc_auc_score += roc_auc_score(y_pred_oneVSrest, y_true_oneVSrest, size);
	}
	avg_roc_auc_score /= n_classes;

	if (y_true_oneVSrest != nullptr) {
		delete[] y_true_oneVSrest;
		y_true_oneVSrest = nullptr;
	}

	return avg_roc_auc_score;
}

float Metrics::pr_auc_score(float* y_pred, int* y_true, int size) {
	int TP = 0, FP = 0, TN = 0, FN = 0;
	int *idx;
	float *x, *y, ret_auc;

	idx = argsort(y_pred, size, DESC);

	// predict all the example to negtive
	for (int i = 0; i < size; i++) {
		if (y_true[idx[i]] == 1) FN++;
		else TN++;
	}

	x = new float[size];
	y = new float[size];
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

	return ret_auc > 1.0 ? 1.0 : ret_auc;
}

float Metrics::pr_auc_score_multi(float* y_pred, int* y_true, int n_classes, int size) {
	float avg_pr_auc_score = 0.0;
	int*y_true_oneVSrest = new int[size];
	float *y_pred_oneVSrest;
	for (int c = 0; c < n_classes; c++) {
		y_pred_oneVSrest = y_pred + c*size;
		for (int i = 0; i < size; i++) {
			y_true_oneVSrest[i] = y_true[i] == c ? 1 : 0;
		}
		avg_pr_auc_score += pr_auc_score(y_pred_oneVSrest, y_true_oneVSrest, size);
	}
	avg_pr_auc_score /= n_classes;

	if (y_true_oneVSrest != nullptr) {
		delete[] y_true_oneVSrest;
		y_true_oneVSrest = nullptr;
	}
	return avg_pr_auc_score;
}

float Metrics::auc(float* x, float* y, int size) {
	int *idx;
	float last_x = 0.0, ret_auc = 0.0;
	
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

void Metrics::performance_report(float* y_pred, int* y_true, int n_test, float threshold) {
	std::string color_info = "yellow", color_value = "red";
	std::cout << "TestSet Size: " << n_test << std::endl;
	std::cout << "Threshold: " << threshold << std::endl;

	std::cout << color_msg("Precision = ", color_info ) << color_msg(Metrics::precision(y_pred, y_true, n_test, threshold), color_value) << std::endl;
    std::cout << color_msg("Recall = ", color_info) << color_msg(Metrics::recall(y_pred, y_true, n_test, threshold), color_value) << std::endl;
    std::cout << color_msg("F1-score = ", color_info) << color_msg(Metrics::f1_score(y_pred, y_true, n_test, threshold), color_value) << std::endl;
    std::cout << color_msg("AUC = ", color_info) << color_msg(Metrics::roc_auc_score(y_pred, y_true, n_test), color_value) << std::endl;
    std::cout << color_msg("Precision-Recall AUC = ", color_info) << color_msg(Metrics::pr_auc_score(y_pred, y_true, n_test), color_value) << std::endl;
}

void Metrics::performance_report(const std::string& filename, float* y_pred, int* y_true, int n_test, float threshold) {
	std::ofstream out;
	out.open(filename.c_str(), std::ios::out);
	if (!out.is_open()) {
		std::cerr << "Fail to open " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	std::string color_info = "yellow", color_value = "red";
	out << "TestSet Size\t" << n_test << std::endl;
	out << "Threshold\t" << threshold << std::endl;

	out << "Precision\t" << Metrics::precision(y_pred, y_true, n_test, threshold) << std::endl;
    out << "Recall\t" << Metrics::recall(y_pred, y_true, n_test, threshold) << std::endl;
    out << "F1-score\t" << Metrics::f1_score(y_pred, y_true, n_test, threshold) << std::endl;
    out << "AUC\t" << Metrics::roc_auc_score(y_pred, y_true, n_test) << std::endl;
    out << "Precision-Recall AUC\t" << Metrics::pr_auc_score(y_pred, y_true, n_test) << std::endl;

	out.close();
}
