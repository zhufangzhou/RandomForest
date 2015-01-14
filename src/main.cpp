#include "dataset.h"
#include "tree.h"
#include "utils.h"
#include "metrics.h"
#include "cmdLine.h"

int main(int argc, char** argv) {
	cmdLineParser cmd(argc, argv);

	const std::string option_help 		= cmd.registerOption("help", "display the help page");
	const std::string option_train 		= cmd.registerOption("train", "filename of training data");
	const std::string option_test 		= cmd.registerOption("test", "filename of testing data (must without labels)");
	const std::string option_validate 	= cmd.registerOption("validation", "filename of validation data (must with labels)");
	const std::string option_dump 		= cmd.registerOption("dump", "directory path for forest model to dump"); 
	const std::string option_load 		= cmd.registerOption("load", "directory path for forest model to load");
	const std::string option_dot 		= cmd.registerOption("dot", "dot filename of the forest");
	const std::string option_imp 		= cmd.registerOption("imp", "`stdout` means print the feature importance on the screen, `filename` means the filename to print importance. IN NO ASCENDING ORDER.");
	const std::string option_imp_n 		= cmd.registerOption("imp_n", "how many top feature importance to print, print all if dont't have this option");	

	/* check command line options */
	cmd.checkOption();

	/* show help information */
	if (argc < 2 || cmd.hasOption(option_help)) {
		cmd.displayOption();
		exit(EXIT_SUCCESS);
	}

	if (cmd.hasOption("train") && cmd.hasOption("load")) {
		std::cerr << color_msg("You can only choose `train` or `load` one at a time.", "red") << std::endl;
		exit(EXIT_FAILURE);
	}
	return 0;
}
