ICX=icx

BIN_DIR=build
SRC_DIR=src

BS_DIR=${SRC_DIR}/baselines
VF_DIR=${BS_DIR}/verification
PERF_DIR=${SRC_DIR}/performance
ROOF_DIR=${SRC_DIR}/roofline
OPT_DIR=${SRC_DIR}/optimizations
MMM_DIR=${SRC_DIR}/mmm
PERF_MMM_DIR=${SRC_DIR}/performance_mmm




# Compile with ICX and then run
icx -I include/ -I /opt/openblas/include -Wall -o build//comp_test_icx \
${BS_DIR}/baseline1/baseline1.c \
${BS_DIR}/baseline2/baseline2.c \
${OPT_DIR}/opt*.c \
${OPT_DIR}/alg*.c \
${OPT_DIR}/optimizations_utils/optimizations_utils.c \
${BS_DIR}/baselines_utils/baselines_utils.c \
${PERF_DIR}/performance.c \
-lm -L/opt/openblas/lib -lopenblas_nonthreaded

./build/comp_test_icc 29 200 1800 25 ../docs/outputs/icx_noflag.out

icx -I include/ -I /opt/openblas/include -Wall -xskylake build//comp_test_icx \
${BS_DIR}/baseline1/baseline1.c \
${BS_DIR}/baseline2/baseline2.c \
${OPT_DIR}/opt*.c \
${OPT_DIR}/alg*.c \
${OPT_DIR}/optimizations_utils/optimizations_utils.c \
${BS_DIR}/baselines_utils/baselines_utils.c \
${PERF_DIR}/performance.c \
-lm -L/opt/openblas/lib -lopenblas_nonthreaded

./build/comp_test_icc 29 200 1800 25 ../docs/outputs/icx_xskylake.out

icx -I include/ -I /opt/openblas/include -Wall -xskylake -ffast-math -fno-signed-zeros -menable-unsafe-fp-math build//comp_test_icx \
${BS_DIR}/baseline1/baseline1.c \
${BS_DIR}/baseline2/baseline2.c \
${OPT_DIR}/opt*.c \
${OPT_DIR}/alg*.c \
${OPT_DIR}/optimizations_utils/optimizations_utils.c \
${BS_DIR}/baselines_utils/baselines_utils.c \
${PERF_DIR}/performance.c \
-lm -L/opt/openblas/lib -lopenblas_nonthreaded

./build/comp_test_icc 29 200 1800 25 ../docs/outputs/icx_all.out