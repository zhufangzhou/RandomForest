#pragma once

/* forest default parameter */
const int DEFAULT_MAX_DEPTH = -1;
const int DEFAULT_MIN_SAMPLE_LEAF = 1;
const int DEFAULT_N_TREES = 10;
const int DEFAULT_N_THREADS = 1;

/* forest export_dotfile parameter */
enum dotfile_mode {SEPARATE_TREES, WHOLE_FOREST};

/* dataset read mode */
enum learn_mode {TRAIN, TEST};
