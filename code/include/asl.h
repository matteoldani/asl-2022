#define EPSILON 0.05
#define MAX_ITERATIONS 100
#define SEED 42
#define RANK 24 //needs to be divisible by BLOCK_SIZE_TRANS, BLOCK_SIZE_MMUL and BLOCK_SIZE_RTRANSMUL
#define M_TEST 400 //needs to be divisible by BLOCK_SIZE_TRANS, BLOCK_SIZE_MMUL and BLOCK_SIZE_RTRANSMUL
#define N_TEST 400 //needs to be divisible by BLOCK_SIZE_TRANS, BLOCK_SIZE_MMUL and BLOCK_SIZE_RTRANSMUL
#define NUM_RUNS 10
#define CLALIBRATE_ITERATIONS 100
#define BLOCK_SIZE_TRANS 4 //needs to be divisible by 4 because of the loop unrolling
#define BLOCK_SIZE_MMUL 4
#define BLOCK_SIZE_RTRANSMUL 4
typedef unsigned long long myInt64;
