/**
 * @file dataset.h
 * @brief 
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-11-18
 */
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <cstring>

#include "utils.h"

typedef short target_t; 	/** label data type */
typedef float feature_t; 	/** feature data type */
enum learn_mode {TRAIN, PREDICT}; 	/** `train` mode or `predict` mode */

typedef struct {
	int ex_id;  /** example id */
	feature_t fea_value; /** feature value */
	void set(int ex_id, feature_t fea_value) {
		this->ex_id = ex_id;
		this->fea_value = fea_value;
	}
}ev_pair_t;

class example_t {
	public:
		target_t y; /** example label*/
		int nnz; 	/** number of non-zero attribute in this example */
		int* fea_id; 	/** array of non-zero feature id */
		feature_t* fea_value; /** array of non-zero feature value */

		/**
		 * @brief example_t constructor
		 */
		example_t();
		/**
		 * @brief ~example_t deconstructor
		 */
		~example_t();
		/**
		 * @brief push_back push an entry to this example
		 *
		 * @param id feature id
		 * @param value feature value
		 */
		void push_back(int id, feature_t value);
		/**
		 * @brief debug print some information for debugging
		 */
		void debug();
};

class DataReader {
	private:
		int n_features;		/** number of features in the input file */
		std::ifstream ifs; 		/** input file stream related to the input file */
		learn_mode mode; 	/** learn mode */
	public:
		/**
		 * @brief DataReader constructor
		 *
		 * @param n_features number of features
		 * @param mode train or predict
		 */
		DataReader(const std::string& filename, int n_features, const learn_mode mode);
		/**
		 * @brief ~DataReader deconstructor
		 */
		~DataReader();
		/**
		 * @brief read_an_example read an example
		 *
		 * @param ifs input file stream to read example
		 *
		 * @return a single example's features
		 */
		example_t* read_an_example();		
		/**
		 * @brief read_examples read all the example
		 *
		 * @param filename 
		 *
		 * @return a vector contains all examples' features
		 */
		std::vector<example_t*> read_examples();
};

class Dataset {
	private:
		ev_pair_t** x; 		/** each row is an attribute */	
		int* size; 			/** number of examples with non-zero feature values for each attribute */
		target_t* y; 		/** label for each example */
		float* weight; 		/** weight for each class */
		
		int n_classes; 		/** number of classes */
		int n_examples;		/** number of examples */
		int n_features; 	/** number of attributes */

		bool* is_cate; 		/** is the ith attribute categorical */

		bool is_init; 		/** boolean variable to indicate whether dataset has been initialized */
		learn_mode mode; 	/** learn mode */
		
		/**
		 * @brief isort code comes from `fest package` http://lowrank.net/nikos/fest/
		 *
		 * @param a example_id-feature_value pair array
		 * @param f corresponding feature_id in array `a`
		 * @param n array length
		 */
		void isort(ev_pair_t* a, int* f, int n);
		/**
		 * @brief qsortlazy code comes from `fest package` http://lowrank.net/nikos/fest/
		 *
		 * @param a example_id-feature_value pair array
		 * @param f corresponding feature_id in array `a`
		 * @param l begin index in array
		 * @param u end index in array
		 */
		void qsortlazy(ev_pair_t* a, int* f, int l, int u);
		/**
		 * @brief sort code comes from `fest package` http://lowrank.net/nikos/fest/
		 *
		 * @param a example_id-feature_value pair array
		 * @param f corresponding feature_id in array `a`
		 * @param len array length
		 */
		void sort(ev_pair_t* a, int* f, int len);
	public:
		/**
		 * @brief Dataset constructor
		 */
		Dataset();
		/**
		 * @brief Dataset constructor
		 *
		 * @param n_classes number of classes in the training set
		 * @param n_features number of features
		 * @param weight weight for each class
		 */
		Dataset(int n_classes, int n_features, float* weight);
		/**
		 * @brief ~Dataset deconstructor
		 */
		~Dataset();
		/**
		 * @brief init 
		 *
		 * @param n_classes number of classes in the training set
		 * @param n_features number of features
		 * @param weight weight for each class
		 */
		void init(int n_classes, int n_features, float* weight);
		/**
		 * @brief load_data generate the dataset from input file
		 *
		 * @param filename input file name
		 * @param mode `TRAIN` or `PREDICT`
		 */
		void load_data(const std::string& filename, const learn_mode mode);
		/**
		 * @brief load_data_meta 
		 *
		 * @param filename
		 */
		void load_data_meta(const std::string& filename);
		/**
		 * @brief debug print some information for debugging
		 */
		void debug();
};

