BIN_DIR := ../bin/
BUILD_DIR := ../build/
MODEL_DIR := ../model/
DISPLAY_DIR := ../display/
INCLUDE_DIR := ../include/
UTILS_DIR := ../../utils/

CC := g++
UTILS_OBJ := ${BUILD_DIR}utils.o ${BUILD_DIR}random.o ${BUILD_DIR}parallel.o
#ALL_OBJ := $(patsubst %.cpp,${BUILD_DIR}%.o, $(wildcard *.cpp)) ${UTILS_OBJ}
ALL_OBJ := ${BUILD_DIR}dataset.o ${BUILD_DIR}tree.o ${BUILD_DIR}forest.o ${BUILD_DIR}metrics.o ${UTILS_OBJ} ${BUILD_DIR}main.o
CXXFLAGS := -g -stdlib=libc++ -std=c++11 -pthread -I${INCLUDE_DIR} -I${UTILS_DIR}include `pkg-config --cflags libconfig` `pkg-config --libs libconfig`

all: create_dir main

main: $(ALL_OBJ)
	$(CC) -g $(ALL_OBJ) -o ${BIN_DIR}$@ 

debug: ${BUILD_DIR}debug.o ${BUILD_DIR}dataset.o ${BUILD_DIR}utils.o ${BUILD_DIR}tree.o ${BUILD_DIR}metrics.o ${BUILD_DIR}random.o ${BUILD_DIR}forest.o ${BUILD_DIR}parallel.o
	g++ $^ -o ${BIN_DIR}$@ -std=c++11 -pthread

${BUILD_DIR}utils.o: ${UTILS_DIR}src/utils.cpp
	g++ -c $^ -o $@ ${CXXFLAGS}
${BUILD_DIR}parallel.o: ${UTILS_DIR}src/parallel.cpp
	g++ -c $^ -o $@ ${CXXFLAGS}
${BUILD_DIR}random.o: ${UTILS_DIR}src/random.cpp
	g++ -c $^ -o $@ ${CXXFLAGS}

${BUILD_DIR}%.o: %.cpp
	g++ -c $< -o $@ ${CXXFLAGS}

.PHONY: create_dir
create_dir:
	@mkdir -p ${BIN_DIR} ${BUILD_DIR} ${MODEL_DIR} ${DISPLAY_DIR}

.PHONY: clean
clean:
	rm -f ${BUILD_DIR}main ${ALL_OBJ} ${BUILD_DIR}utils.o ${BUILD_DIR}parallel.o ${BUILD_DIR}random.o ${BUILD_DIR}debug.o ${BIN_DIR}debug 
