define newline


endef

CC=gcc 
CLANG=clang
ICX=icx
ICC=icc

IDIR_LOCALC=./include 
IDIR_BLAS=/opt/openblas/include

BIN_DIR=./build
SRC_DIR=./src
TESTS_DIR=./tests

BS_DIR=$(SRC_DIR)/baselines
VF_DIR=$(BS_DIR)/verification
PERF_DIR=$(SRC_DIR)/performance
ROOF_DIR=$(SRC_DIR)/roofline
OPT_DIR=$(SRC_DIR)/optimizations
MMM_DIR=$(SRC_DIR)/mmm
PERF_MMM_DIR=$(SRC_DIR)/performance_mmm


CINCLUDE=-I $(IDIR_LOCALC)
CINCLUDE+=-I $(IDIR_BLAS)

CFLAGS=-Wall
CFLAGS+=-O3 -mfma
CFLAGS+=-g 
CFLAGS+=-fno-tree-vectorize
#CFLAGS+=-ffast-math
CFLAGS+=-march=native


FLAG_LINK=-lm
FLAG_LINK+=-L/opt/openblas/lib
FLAG_LINK+=-lopenblas_nonthreaded

MKLROOT=/opt/intel/oneapi/mkl/latest

# INTEL_LINK=${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -lpthread -lm -ldl
# INTEL_FLAG= -DMKL_ILP64  -m64  -I"$(MKLROOT)/include"


all: baseline performance optimisations tests

.build_dir_creation:
	mkdir -p $(BIN_DIR)

baseline: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/baseline \
	$(BS_DIR)/baseline1/baseline1.c \
	$(BS_DIR)/baseline2/baseline2.c \
	$(BS_DIR)/baselines_utils/baselines_utils.c \
	$(BS_DIR)/run_baseline.c \
	$(FLAG_LINK) $(INTEL_LINK) $(INTEL_FLAG)

performance: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/performance \
	$(BS_DIR)/baseline1/baseline1.c \
	$(BS_DIR)/baseline2/baseline2.c \
	$(OPT_DIR)/opt*.c \
	$(OPT_DIR)/alg*.c \
	$(OPT_DIR)/optimizations_utils/optimizations_utils.c \
	$(BS_DIR)/baselines_utils/baselines_utils.c \
	$(PERF_DIR)/performance.c \
	$(FLAG_LINK) $(INTEL_LINK) $(INTEL_FLAG)

performance_icx: .build_dir_creation
	$(ICX) $(CINCLUDE) -march=native -o $(BIN_DIR)/performance_icx \
	$(BS_DIR)/baseline1/baseline1.c \
	$(BS_DIR)/baseline2/baseline2.c \
	$(OPT_DIR)/opt*.c \
	$(OPT_DIR)/alg*.c \
	$(OPT_DIR)/optimizations_utils/optimizations_utils.c \
	$(BS_DIR)/baselines_utils/baselines_utils.c \
	$(PERF_DIR)/performance.c \
	$(FLAG_LINK) $(INTEL_LINK) $(INTEL_FLAG)


roofline: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/roofline \
	$(BS_DIR)/baseline1/baseline1.c \
	$(BS_DIR)/baseline2/baseline2.c \
	$(OPT_DIR)/opt*.c \
	$(OPT_DIR)/alg*.c \
	$(OPT_DIR)/optimizations_utils/optimizations_utils.c \
	$(BS_DIR)/baselines_utils/baselines_utils.c \
	$(ROOF_DIR)/roofline.c \
	$(FLAG_LINK) $(INTEL_LINK) $(INTEL_FLAG)

tests: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/tests \
	$(BS_DIR)/baseline1/baseline1.c \
	$(BS_DIR)/baseline2/baseline2.c \
	$(BS_DIR)/baselines_utils/baselines_utils.c \
	$(OPT_DIR)/optimizations_utils/optimizations_utils.c \
	$(OPT_DIR)/opt*.c \
	$(OPT_DIR)/alg*.c \
	$(TESTS_DIR)/tests.c $(FLAG_LINK) $(INTEL_LINK) $(INTEL_FLAG)

optimisations: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/optimisations $(OPT_DIR)/*.c  $(FLAG_LINK) $(INTEL_LINK) $(INTEL_FLAG)



least_square: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/least_square $(BS_DIR)/least_square.c $(FLAG_LINK)

mmm: .build_dir_creation
	$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/mmm \
	$(BS_DIR)/baselines_utils/baselines_utils.c \
	$(MMM_DIR)/*.c \
	$(FLAG_LINK) $(INTEL_LINK) $(INTEL_FLAG)

performance_mmm: .build_dir_creation
				$(CC) $(CINCLUDE) $(CFLAGS) -o $(BIN_DIR)/performance_mmm \
				$(BS_DIR)/baselines_utils/baselines_utils.c \
				$(MMM_DIR)/mmm_*.c \
				$(PERF_MMM_DIR)/*.c\
				$(FLAG_LINK) $(INTEL_LINK) $(INTEL_FLAG)

clean:
	rm -r $(BIN_DIR)/*

