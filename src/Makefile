BIN_DIR := ../bin/
BUILD_DIR := ../build/
MODEL_DIR := ../model/
INCLUDE_DIR := ../include/
UTILS_DIR := ../../utils/

CC := g++
ALL_OBJ := $(patsubst %.cpp,${BUILD_DIR}%.o, $(wildcard *.cpp)) utils.o
CXXFLAGS := -g -Wno-write-strings -std=c++0x -I${INCLUDE_DIR} -I${UTILS_DIR}include

all: create_dir main

main: $(ALL_OBJ)
	$(CC) $(CXXFLAGS) $(ALL_OBJ) -o main

debug: ${BUILD_DIR}debug.o ${BUILD_DIR}dataset.o ${BUILD_DIR}utils.o ${BUILD_DIR}tree.o ${BUILD_DIR}metrics.o ${BUILD_DIR}random.o ${BUILD_DIR}forest.o ${BUILD_DIR}parallel.o
	g++ $^ -o ${BIN_DIR}$@

${BUILD_DIR}utils.o: ${UTILS_DIR}src/utils.cpp
	g++ -g -std=c++0x -c $^ -o $@ -I${UTILS_DIR}include
${BUILD_DIR}parallel.o: ${UTILS_DIR}src/parallel.cpp
	g++ -g -std=c++0x -c $^ -o $@ -I${UTILS_DIR}include
${BUILD_DIR}random.o: ${UTILS_DIR}src/random.cpp
	g++ -g -std=c++0x -c $^ -o $@ -I${UTILS_DIR}include

${BUILD_DIR}%.o: %.cpp
	g++ -g -std=c++0x -c $< -o $@ -I${INCLUDE_DIR} -I${UTILS_DIR}include

.PHONY: create_dir
create_dir:
	@mkdir -p ${BIN_DIR} ${BUILD_DIR} ${MODEL_DIR}

.PHONY: clean
clean:
	rm -f ${BUILD_DIR}main ${ALL_OBJ} ${BIN_DIR}debug
