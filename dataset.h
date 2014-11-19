/**
 * @file dataset.h
 * @brief 
 * @author Zhu Fangzhou, zhu.ark@gmail.com
 * @version 1.0
 * @date 2014-11-18
 */
#ifndef __DATASET_HEADER
#define __DATASET_HEADER

#include <string>

typedef short target_t;
typedef float feature_t;
enum learn_mode {TRAIN, PREDICT};

typedef struct {
	int ex_id;  /** example id */
	feature_t fea_value; /** feature value */
}ev_pair_t;

class example_t {
	public:
		target_t y;
		int nnz;
		int* fea_id;
		feature_t* fea_value;
		example_t();
		~example_t();
		void push_back(int id, feature_t value);
};

class DataReader {
	private:
		int n_features;	
		std::ifstream ifs;
	public:
		/**
		 * @brief DataReader constructor
		 *
		 * @param n_features number of features
		 */
		DataReader(std::string filename, int n_features);
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
		 * @return a sparse matrix contains all examples' features
		 */
		ev_pair_t** read_examples();
};

class Dataset {
	private:
		ev_pair_t** x; 		/** each row is an attribute */	
		int* size; 			/** number of examples with non-zero feature value for each attribute */
		target_t* y; 			/** label for each example */
		
		int n_classes; 		/** number of classes */
		int n_examples;		/** number of examples */
		int n_features; 	/** number of attributes */

		bool* is_cate; 		/** is the ith attribute categorical */
	public:
		/**
		 * @brief load_data 
		 *
		 * @param filename
		 * @param mode
		 */
		void load_data(std::string& filename, learn_mode mode);
};

#endif
