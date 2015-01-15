#include <iostream>
#include <string>
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
	int max_depth, min_sample_leaf, n_trees, n_threads;
	float* weight = nullptr;
	libconfig::Config cfg;
	cmdLineParser cmd(argc, argv);

	const std::string option_help 				= cmd.registerOption("help", "show the help page");
	const std::string option_config 			= cmd.registerOption("config", "filename of configure");
	const std::string option_train 				= cmd.registerOption("train", "filename of training data");
	const std::string option_test 				= cmd.registerOption("test", "filename of testing data (must without labels)");
	const std::string option_validate 			= cmd.registerOption("validation", "filename of validation data (must with labels)");
	const std::string option_dump 				= cmd.registerOption("dump", "directory path for forest model to dump"); 
	const std::string option_load 				= cmd.registerOption("load", "directory path for forest model to load");
	const std::string option_dot 				= cmd.registerOption("dot", "dot filename of the forest");
	const std::string option_imp 				= cmd.registerOption("imp", "`stdout` means print the feature importance on the screen, `filename` means the filename to print importance. IN NO ASCENDING ORDER.");
	const std::string option_imp_n 				= cmd.registerOption("imp_n", "how many top feature importance to print, print all if dont't have this option");	
	const std::string option_max_depth 			= cmd.registerOption("max_depth", "the max depth of each decision tree");
	const std::string option_min_sample_leaf 	= cmd.registerOption("min_sample_leaf", "minimum samples in each leaf");
	const std::string option_n_trees 			= cmd.registerOption("n_trees", "number of trees in the forest");
	const std::string option_n_threads 			= cmd.registerOption("n_threads", "number of threads when training or predicting");

	/* check command line options */
	cmd.checkOption();


	/* show help information */
	if (argc < 2 || cmd.hasOption(option_help)) {
		cmd.displayOption();
		exit(EXIT_SUCCESS);
	}

	/* check if have configure file */
	if (!cmd.hasOption(option_config)) {
		cfg.readFile("conf/rf.cfg");
	} else {
		cfg.readFile(cmd.getOptionValue(option_config).c_str());
	}
	const libconfig::Setting& root = cfg.getRoot();
	const libconfig::Setting& dataset = root["Dataset"];
	int n_features = 0;
	dataset.lookupValue("n_features", n_features);
	std::cout << n_features << std::endl;
	return 0;

	if (cmd.hasOption(option_train) && cmd.hasOption(option_load)) {
		std::cerr << error_msg("You can only choose `train` or `load` one at a time.") << std::endl;
		exit(EXIT_FAILURE);
	} else if (!cmd.hasOption(option_train) && !cmd.hasOption(option_load)) {
		std::cerr << error_msg("You must specify the source of model, from training or model file trained before.");
		exit(EXIT_FAILURE);
	} else {
		/* model is from training */
		if (cmd.hasOption(option_train)) {
			if (cmd.hasOption(option_max_depth))
				max_depth = atoi(cmd.getOptionValue(option_max_depth).c_str());
			else 
				max_depth = -1;

			if (cmd.hasOption(option_min_sample_leaf))
				min_sample_leaf = atoi(cmd.getOptionValue(option_min_sample_leaf).c_str());
			else
				min_sample_leaf = 1;

			if (cmd.hasOption(option_n_trees)) 
				n_trees = atoi(cmd.getOptionValue(option_n_trees).c_str());
			else 
				n_trees = 10;

			if (cmd.hasOption(option_n_threads))
				n_threads = atoi(cmd.getOptionValue(option_n_threads).c_str());
			else 
				n_threads = 1;

		} else { /* model is from model file trained before */

		}
	}

	if (weight != nullptr) {
		delete weight;
		weight = nullptr;
	}
	return 0;
}
