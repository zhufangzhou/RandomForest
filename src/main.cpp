#include <iostream>
#include <string>
#include <sstream>
#include <libconfig.h++>

#include "dataset.h"
#include "tree.h"
#include "utils.h"
#include "metrics.h"
#include "cmdLine.h"

std::string error_msg(std::string msg) {
	return color_msg(msg, "red");
}

int main(int argc, char** argv) {
	int max_depth, min_sample_leaf, n_trees, n_threads, n_classes, n_features;
	std::string config_filename, criterion, train_path, input_model_path;
	float* weight = nullptr;
	libconfig::Config cfg;
	dataset *d = nullptr;
	random_forest_classifier *rf = nullptr;

	/* parse command line argument */
	cmdLineParser cmd(argc, argv);

	/* register parameters */
	const std::string option_help 				= cmd.registerOption("help", "show the help page");
	const std::string option_config 			= cmd.registerOption("config", "filename of configure");
	const std::string option_train 				= cmd.registerOption("train", "add this option means train new model");
	const std::string option_test 				= cmd.registerOption("test", "add this option means predict datasets and get labels");
	const std::string option_validate 			= cmd.registerOption("validation", "add this option means measure the performance of the model");
	const std::string option_dump 				= cmd.registerOption("dump", "directory path for forest model to dump"); 
	const std::string option_load 				= cmd.registerOption("load", "directory path for forest model to load");
	const std::string option_dot 				= cmd.registerOption("dot", "dot filename of the forest");
	const std::string option_imp 				= cmd.registerOption("imp", "`stdout` means print the feature importance on the screen, `filename` means the filename to print importance. IN NO ASCENDING ORDER.");
	const std::string option_imp_n 				= cmd.registerOption("imp_n", "how many top feature importance to print, print all if dont't have this option");	

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
			config_filename = "conf/rf.cfg.template";
		} else {
			config_filename = cmd.getOptionValue(option_config);
		}
		cfg.readFile(config_filename.c_str());

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
				const libconfig::Setting& random_forest_cfg = root["RandomForests"];
				/* read the hyperparameters, if do not appear in configure file then set the default values */
				if (!random_forest_cfg.lookupValue("n_trees", n_trees)) n_trees = 10;
				if (!random_forest_cfg.lookupValue("n_threads", n_threads)) n_threads = 1;
				if (!random_forest_cfg.lookupValue("max_depth", max_depth)) max_depth = -1;
				if (!random_forest_cfg.lookupValue("min_sample_leaf", min_sample_leaf)) min_sample_leaf = 1;
				if (!random_forest_cfg.lookupValue("criterion", criterion)) criterion = "sqrt";

				const libconfig::Setting& train_data_cfg = root["Train_Dataset"];
				if (!train_data_cfg.lookupValue("path", train_path)) {
					std::cerr << error_msg("You must specify a training dataset path under `Train_Dataset` in your configure file.");
					exit(EXIT_FAILURE);
				}
				if (!train_data_cfg.lookupValue("n_classes", n_classes)) {
					std::cerr << error_msg("You must give `n_classes` under `Train_Dataset` in your configure file.");
					exit(EXIT_FAILURE);
				}
				if (!train_data_cfg.lookupValue("n_features", n_features)) {
					std::cerr << error_msg("You must give `n_features` under `Train_Dataset` in your configure file.");
					exit(EXIT_FAILURE);
				}
				/* allocate space to weight vector */
				weight = new float[n_classes];
				std::string weight_str;
				if (!train_data_cfg.lookupValue("weight", weight_str)) {
					for (int c = 0; c < n_classes; c++) weight[c] = 1.0 / n_classes;
				} else {
					/* HAVE NOT CHECK ERROR HERE !!!!!!!! */
					std::stringstream ss(weight_str);
					std::string weight_entry;
					int c = 0;
					float weight_tot = 0.0;
					while (std::getline(ss, weight_entry, ',') && c < n_classes) {
						weight[c] = atof(weight.c_str());
						weight_tot += weight[c];
						c++;
					}
					/* normalize */
					for (c = 0; c < n_classes; c++) weight[c] /= weight_tot;
				}

				/* create dataset object */
				d = new dataset(n_classes, n_features, weight);
				d->load_data(path, TRAIN);

				/* create random forest classifier object */
				rf = new random_forest_classifier(criterion, max_depth, min_sample_leaf, n_trees, n_threads);

				/* build forest */
				rf->build(d);
			} else { /* model is from model file trained before */
				const libconfig::Setting& input_model_cfg = root["Input_Model"];
				/* create random forest classifier object */
				rf = new random_forest_classifier();

				if (!input_model_cfg.lookupValue("path", input_model_path)) {
					std::cerr << error_msg("You must give `path` under `Input_Model` in your configure file.") << std::endl;
					exit(EXIT_FAILURE);
				}

				/* load the model */
				rf->load(input_model_path);
			}
		}


		// TODO: predict validate ....


		if (weight != nullptr) {
			delete weight;
			weight = nullptr;
		}
	} catch (const libconfig::SettingTypeException& e) {
		std::cerr << e.what() << std::endl;	
	} catch (const libconfig::SettingNotFoundException& e) {
		std::cerr << e.what() << std::endl;
	} catch (const libconfig::SettingNameException& e) {
		std::cerr << e.what() << std::endl;
	} catch (const libconfig::ParseException& e) {
		std::cerr << "Parse error at" << e.getFile() << ":" e.getLine()
				  << "-" << e.getError() << std::endl;
	} catch (const libconfig::FileIOException& e) {
		std::cerr << e.what() << std::endl;
	}
	return 0;
}
