/* C++ header files */
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

/* libconfig header files */
#include <libconfig.h++>

/* my header files */
#include "dataset.h"
#include "tree.h"
#include "forest.h"
#include "utils.h"
#include "metrics.h"
#include "cmdLine.h"
#include "constant.h"

std::string error_msg(std::string msg) {
	return color_msg(msg, "red");
}

int main(int argc, char** argv) {
	int max_depth, min_sample_leaf, n_trees, n_threads, n_classes, n_features;
	std::string config_path, criterion, train_path, test_path, validate_path, input_model_path, output_model_path, dot_file_path;
	float* weight = nullptr;
	libconfig::Config cfg;
	dataset *d = nullptr;
	random_forest_classifier *rf = nullptr;

	/* parse command line argument */
	cmdLineParser cmd(argc, argv);

	/* register parameters */
	const std::string option_help 		= cmd.registerOption("help", "show the help page");
	const std::string option_config 	= cmd.registerOption("config", "filename of configure");
	const std::string option_train 		= cmd.registerOption("train", "add this option means train new model");
	const std::string option_test 		= cmd.registerOption("test", "add this option means predict datasets and get labels");
	const std::string option_validate 	= cmd.registerOption("validation", "add this option means measure the performance of the model");
	const std::string option_dump 		= cmd.registerOption("dump", "add this option means to dump trained model"); 
	const std::string option_load 		= cmd.registerOption("load", "add this option means model is load from file");
	const std::string option_dot 		= cmd.registerOption("dot", "add this option means to generate dot file");

	/* check command line options */
	cmd.checkOption();


	/* show help information */
	if (argc < 2 || cmd.hasOption(option_help)) {
		cmd.displayOption();
		exit(EXIT_SUCCESS);
	}

	try {
		/* check if have configure file */
		if (!cmd.hasOption(option_config)) { /* if do not give configure file, then read the default one */
			config_path = "conf/rf.cfg.template";
		} else {
			config_path = cmd.getOptionValue(option_config);
		}
		cfg.readFile(config_path.c_str());

		/* get setting objects from configure file */
		const libconfig::Setting& root = cfg.getRoot();

		/* generate model by training or loading */
		if (cmd.hasOption(option_train) && cmd.hasOption(option_load)) {
			std::cerr << error_msg("You can only choose `train` or `load` one at a time.") << std::endl;
			exit(EXIT_FAILURE);
		} else if (!cmd.hasOption(option_train) && !cmd.hasOption(option_load)) {
			std::cerr << error_msg("You must specify the source of model, from training or model file trained before.");
			exit(EXIT_FAILURE);
		} else {
			/* model is from training */
			if (cmd.hasOption(option_train)) {
				const libconfig::Setting& random_forest_cfg = root["RandomForest"];
				/* read the hyperparameters, if do not appear in configure file then set the default values */
				if (!random_forest_cfg.lookupValue("n_trees", n_trees)) n_trees = DEFAULT_N_TREES;
				if (!random_forest_cfg.lookupValue("n_threads", n_threads)) n_threads = DEFAULT_N_THREADS;
				if (!random_forest_cfg.lookupValue("max_depth", max_depth)) max_depth = DEFAULT_MAX_DEPTH;
				if (!random_forest_cfg.lookupValue("min_sample_leaf", min_sample_leaf)) min_sample_leaf = DEFAULT_MIN_SAMPLE_LEAF;
				if (!random_forest_cfg.lookupValue("criterion", criterion)) criterion = "sqrt";

				const libconfig::Setting& train_cfg = root["Train"];
				if (!train_cfg.lookupValue("path", train_path)) {
					std::cerr << error_msg("You must specify a training dataset path under `Train` in your configure file.");
					exit(EXIT_FAILURE);
				}
				if (!train_cfg.lookupValue("n_classes", n_classes)) {
					std::cerr << error_msg("You must give `n_classes` under `Train` in your configure file.");
					exit(EXIT_FAILURE);
				}
				if (!train_cfg.lookupValue("n_features", n_features)) {
					std::cerr << error_msg("You must give `n_features` under `Train` in your configure file.");
					exit(EXIT_FAILURE);
				}
				/* allocate space to weight vector */
				weight = new float[n_classes];
				std::string weight_str;
				if (!train_cfg.lookupValue("weight", weight_str)) {
					/* if do not set weight, then set each class weight equally */
					for (int c = 0; c < n_classes; c++) weight[c] = 1.0 / n_classes;
				} else {
					/* HAVE NOT CHECK ERROR HERE !!!!!!!! */
					std::stringstream ss(weight_str);
					std::string weight_entry;
					int c = 0;
					float weight_tot = 0.0;
					while (std::getline(ss, weight_entry, ',') && c < n_classes) {
						weight[c] = atof(weight_entry.c_str());
						weight_tot += weight[c];
						c++;
					}
					/* normalize */
					for (c = 0; c < n_classes; c++) weight[c] /= weight_tot;
				}

				/* create dataset object */
				d = new dataset(n_classes, n_features, weight);
				d->load_data(train_path, TRAIN);

				/* create random forest classifier object */
				rf = new random_forest_classifier(criterion, max_depth, min_sample_leaf, n_trees, n_threads);

				/* build forest */
				rf->build(d);
			} else { /* model is from model file trained before */
				const libconfig::Setting& input_model_cfg = root["Input_Model"];
				/* create random forest classifier object */
				rf = new random_forest_classifier();

				if (!input_model_cfg.lookupValue("path", input_model_path)) {
					std::cerr << error_msg("You must give `path` under `Input_Model` in your configure file if you want to load model.") << std::endl;
					exit(EXIT_FAILURE);
				}

				/* load the model */
				rf->load(input_model_path);
			}
		}

		/* dump model */
		if (cmd.hasOption(option_dump) && cmd.getOptionValue(option_dump) == "1"
				&& !cmd.hasOption(option_load)) {
			const libconfig::Setting& output_model_cfg = root["Output_Model"];

			if (!output_model_cfg.lookupValue("path", output_model_path)) {
				std::cerr << error_msg("You must give `path` under `Output_Model` in your configure file if you want to dump model.") << std::endl;
				exit(EXIT_FAILURE);
			}

			/* dump the model */
			rf->dump(output_model_path);
		}

		/* draw forest (generate dot file) */
		if (cmd.hasOption(option_dot) && cmd.getOptionValue(option_dot) == "1") {
			const libconfig::Setting& random_forest_cfg = root["RandomForest"];

			if (!random_forest_cfg.lookupValue("dot_file_path", dot_file_path)) {
				std::cerr << error_msg("You must give `dot_file_path` under `RandomForest` in your configure if you want to generate dot file.") << std::endl;
				exit(EXIT_FAILURE);
			}

			/* generate dot file */
			rf->export_dotfile(dot_file_path, WHOLE_FOREST);
		}

		/* validate */
		if (cmd.hasOption(option_validate) && cmd.getOptionValue(option_validate) == "1") {
			const libconfig::Setting& validate_cfg = root["Validate"];	
			
			if (!validate_cfg.lookupValue("path", validate_path)) {
				std::cerr << error_msg("You must give `path` under `Validate` in your configure file if you want to validate your model") << std::endl;
				exit(EXIT_FAILURE);
			}
			
			int n_features_v, n_classes_v; 
			if (!validate_cfg.lookupValue("n_features", n_features_v)) {
				std::cerr << error_msg("To avoid mistake, please give `n_features` under `Validate` in your configure file.") << std::endl;
				exit(EXIT_FAILURE);
			}
			if (!validate_cfg.lookupValue("n_classes", n_classes_v)) {
				std::cerr << error_msg("To avoid mistake, please give `n_classes` under `Validate` in your configure file.") << std::endl;
				exit(EXIT_FAILURE);
			}

			n_classes = rf->get_n_classes();
			if (n_classes_v != n_classes) {
				std::cerr << "`n_classes` in `Validate` is " << n_classes_v << ", but which in the model is " << n_classes << "." << std::endl;
				exit(EXIT_FAILURE);
			}

			n_features = rf->get_n_features();
			if (n_features_v != n_features) {
				std::cerr << "`n_features` in `Validate` is " << n_features_v << ", but which in the model is " << n_features << "." << std::endl;
				exit(EXIT_FAILURE);
			}

			/* read validate data */
			data_reader* dr = new data_reader(validate_path, n_features_v, TRAIN);
			std::vector<example_t*> validate_data = dr->read_examples();

			/* validate model */
			int n_validate = validate_data.size();	
			float* proba_all = rf->predict_proba(validate_data);
			float* proba = proba_all + n_validate;
			int* label = new int[n_validate];
			for (int i = 0; i < n_validate; i++) {
				label[i] = validate_data[i]->y;
			}
			
			/* output performance report */
			std::string report_path;
			float threshold = 0.5;
			if (!validate_cfg.lookupValue("report_path", report_path)) {
				/* if do not set `report_path`, print report to screen */
				std::cout << "===============================================" << std::endl;
				std::cout << "=============  Performance Report =============" << std::endl;
				std::cout << "===============================================" << std::endl;
				Metrics::performance_report(proba, label, n_validate, threshold);
			} else {
				/* if set `report_path`, then print report to file */
				Metrics::performance_report(report_path, proba, label, n_validate, threshold);
			}

			/* free space */
			if (dr != nullptr) {
				delete dr;
				dr = nullptr;
			}
			if (label != nullptr) {
				delete[] label;
				label = nullptr;
			}
		}

		/* test */
		if (cmd.hasOption("test") && cmd.getOptionValue("test") == "1") {
			libconfig::Setting& test_cfg = root["Test"];
			
			if (!test_cfg.lookupValue("path", test_path)) {
				std::cerr << error_msg("You must give `path` under `Test` in your configure file.") << std::endl;
				exit(EXIT_FAILURE);
			}

			/* read dataset parameters */
			int n_features_t, n_classes_t; 
			if (!test_cfg.lookupValue("n_features", n_features_t)) {
				std::cerr << error_msg("To avoid mistake, please give `n_features` under `Test` in your configure file.") << std::endl;
				exit(EXIT_FAILURE);
			}
			if (!test_cfg.lookupValue("n_classes", n_classes_t)) {
				std::cerr << error_msg("To avoid mistake, please give `n_classes` under `Test` in your configure file.") << std::endl;
				exit(EXIT_FAILURE);
			}
			/* check dataset parameters whether are match which model */
			n_classes = rf->get_n_classes();
			if (n_classes_t != n_classes) {
				std::cerr << "`n_classes` in `Test` is " << n_classes_t << ", but which in the model is " << n_classes << "." << std::endl;
				exit(EXIT_FAILURE);
			}
			n_features = rf->get_n_features();
			if (n_features_t != n_features) {
				std::cerr << "`n_features` in `Test` is " << n_features_t << ", but which in the model is " << n_features << "." << std::endl;
				exit(EXIT_FAILURE);
			}

			/* read test data */
			data_reader* dr = new data_reader(test_path, n_features_t, TEST);
			std::vector<example_t*> test_data = dr->read_examples();

			/* validate model */
			int n_test = test_data.size();	
			float* proba_all = rf->predict_proba(test_data);
			float* proba = proba + n_test;

			/* read result path */
			std::string result_path;
			if (!test_cfg.lookupValue("result_path", result_path)) {
				std::cerr << error_msg("You must give `result_path` under `Test` in your configure file.") << std::endl;
				exit(EXIT_FAILURE);
			}
			/* read result mode */
			std::string result_mode;
			if (!test_cfg.lookupValue("result_mode", result_mode)) {
				/* if not set, use proba as default */
				result_mode = "proba";
			}
			/* check result mode */
			if (result_mode != "proba" && result_mode != "label") {
				std::cerr << error_msg("Bad Value! The valid value of `result_mode` under `Test` in your configure file are `proba` and `label`.") << std::endl;
				exit(EXIT_FAILURE);
			}

			/* sort the result? */
			std::string sort_mode;
			int* idx = nullptr;
			/* if do not set the sort option, then just skip this step and set idx to natural order */
			if (test_cfg.lookupValue("sort", sort_mode)) {
				/* check sort mode */
				if (!(sort_mode == "asc") && !(sort_mode == "desc")) {
					std::cerr << "Bad `sort_mode` value. Valid values are `asc` and `desc`." << std::endl;
					exit(EXIT_FAILURE);
				}
				/* sort the result and return index in order */
				if (sort_mode == "asc") {
					idx = argsort(proba, n_test, ASC);		
				} else {
					idx = argsort(proba, n_test, DESC);
				}
			} else {
				idx = new int[n_test];
				for (int i = 0; i < n_test; i++) idx[i] = i;
			} // end sort

			std::ofstream out;
			out.open(result_path);
			if (!out.is_open()) {
				std::cerr << "Fail to open result file " << result_path << "." << std::endl;
				exit(EXIT_FAILURE);
			}

			if (result_mode == "proba") {
				/* set format */
				out << std::fixed << std::setprecision(3);
				/* Output Format (assume sort the result descend):
				 * id 	proba
				 * 4 	0.987
				 * 5 	0.932
				 * 1 	0.875
				 * 0 	0.352
				 * 3 	0.298
				 * 2 	0.256
				 * */
				for (int i = 0; i < n_test; i++) {
					out << idx[i] << "\t" << proba[idx[i]] << std::endl;
				}
			} else { /* result_mode = "label" */
				/* read threshold */	
				float threshold;
				if (!test_cfg.lookupValue("threshold", threshold)) {
					/* if not set threshold in the configure file, then set `0.5` as default */
					threshold = 0.5;
				}
				
				/* set format */
				out << std::fixed << std::setprecision(3);
				for (int i = 0; i < n_test; i++) {
					if (proba[idx[i]] >= threshold) {
						out << idx[i] << "\t1" << std::endl;
					} else {
						out << idx[i] << "\t0" << std::endl;
					}
				}
			} // end result_mode

			/* close output stream */
			if (out.is_open()) {
				out.close();
			}
			/* free space */
			if (idx != nullptr) {
				delete[] idx;
				idx = nullptr;
			}
		}
		
		/* free space */
		if (weight != nullptr) {
			delete[] weight;
			weight = nullptr;
		}
	} catch (const libconfig::SettingTypeException& e) {
		std::cerr << e.what() << std::endl;	
	} catch (const libconfig::SettingNotFoundException& e) {
		std::cerr << e.what() << std::endl;
	} catch (const libconfig::SettingNameException& e) {
		std::cerr << e.what() << std::endl;
	} catch (const libconfig::ParseException& e) {
		std::cerr << "Parse error at" << e.getFile() << ":" << e.getLine()
				  << "-" << e.getError() << std::endl;
	} catch (const libconfig::FileIOException& e) {
		std::cerr << e.what() << std::endl;
	}
	return 0;
}
