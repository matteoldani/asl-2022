define newline


endef

CC=gcc 
IDIR_LOCALC=./include 
IDIR_BLAS=/opt/openblas/include

BIN_DIR=./build
SRC_DIR=./src
TESTS_DIR=./tests

BS_DIR=$(SRC_DIR)/baselines
VF_DIR=$(BS_DIR)/verification
PERF_DIR=$(SRC_DIR)/performance
OPT_DIR=$(SRC_DIR)/optimizations

CINCLUDE=-I $(IDIR_LOCALC)
CINCLUDE+=-I $(IDIR_BLAS)

CFLAGS=-Wall
CFLAGS+=-O3
CFLAGS+=-g 
CFLAGS+=-fno-tree-vectorize


FLAG_LINK=-L/opt/openblas/lib
FLAG_LINK+=-lopenblas_nonthreaded
FLAG_LINK+=-lm

all: baseline performance verification optimisations tests

.build_dir_creation:
	mkdir -p $(BIN_DIR)

baseline: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/baseline $(BS_DIR)/baseline1/baseline1.c $(BS_DIR)/baseline2/baseline2.c $(BS_DIR)/baselines_utils/baselines_utils.c $(BS_DIR)/run_baseline.c $(FLAG_LINK)

performance: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/performance \
	$(BS_DIR)/baseline1/baseline1.c \
	$(BS_DIR)/baseline2/baseline2.c \
	$(OPT_DIR)/opt*.c \
	$(OPT_DIR)/alg*.c \
	$(BS_DIR)/baselines_utils/baselines_utils.c \
	$(PERF_DIR)/performance.c $(FLAG_LINK)

verification: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/verification $(BS_DIR)/baseline1/baseline1.c $(BS_DIR)/baseline2/baseline2.c $(BS_DIR)/baselines_utils/baselines_utils.c $(VF_DIR)/verify.c $(FLAG_LINK)

tests: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/tests \
	$(BS_DIR)/baseline1/baseline1.c \
	$(BS_DIR)/baseline2/baseline2.c \
	$(BS_DIR)/baselines_utils/baselines_utils.c \
	$(OPT_DIR)/opt*.c \
	$(OPT_DIR)/alg*.c \
	$(TESTS_DIR)/tests.c $(FLAG_LINK)

optimisations: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/optimisations $(OPT_DIR)/*.c $(FLAG_LINK)



least_square: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/least_square $(BS_DIR)/least_square.c $(FLAG_LINK)

opt1: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/opt1 $(OPT_DIR)/opt1.c $(FLAG_LINK)

optimization_baseline: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/optimization_baseline $(OPT_DIR)/optimization_baseline.c $(FLAG_LINK)




clean:
	rm -r $(BIN_DIR)/*

