#define EPSILON 0.05
#define MAX_ITERATIONS 100
#define SEED 42
#define RANK 16 //needs to be divisible by BLOCK_SIZE_TRANS, BLOCK_SIZE_MMUL and BLOCK_SIZE_RTRANSMUL
#define M_TEST 400 //needs to be divisible by BLOCK_SIZE_TRANS, BLOCK_SIZE_MMUL and BLOCK_SIZE_RTRANSMUL
#define N_TEST 400 //needs to be divisible by BLOCK_SIZE_TRANS, BLOCK_SIZE_MMUL and BLOCK_SIZE_RTRANSMUL
#define NUM_RUNS 10
#define BLOCK_SIZE_NNMF 4
#define CLALIBRATE_ITERATIONS 100
#define BLOCK_SIZE_TRANS 8 //needs to be divisible by 4 because of the loop unrolling
#define BLOCK_SIZE_MMUL 80
#define BLOCK_SIZE_RTRANSMUL 8
#define BLOCK_SIZE_TRANS 8
#define BLOCK_SIZE_H_ROW 8
#define BLOCK_SIZE_H_COL 4
#define BLOCK_SIZE_H_MUL 2
#define BLOCK_SIZE_W 8
#define BLOCK_SIZE_W_ROW 4
#define BLOCK_SIZE_W_COL 8
#define BLOCK_SIZE_W_MUL 2
typedef unsigned long long myInt64;
