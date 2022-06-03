#include <mmm/mmm_0.h>
#include <mmm/mmm_1.h>
#include <mmm/mmm_2.h>
#include <mmm/mmm_3.h>

#include <asl.h>

#include <performance/performance.h>
#include <baselines/baselines_utils.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CPU_FREQ 3.5e9

typedef void (*mmm_m)(Matrix *, Matrix *, Matrix *);

typedef void (*mmm_v)(double *, int, int, double *, int, int, double *, int, int);

mmm_m run_mmm_m;
mmm_v run_mmm_v;

static int double_size = sizeof(double);

// NOTE THAT IS WORKING BECAUSE BLOCK SIZE HARDOCDED TO 1
void transpose(double *src, double *dst, const int N, const int M) {

    int nB = 1;
    int nBM = nB * M;
    int src_i = 0, src_ii;

    for (int i = 0; i < N; i += nB) {
        for (int j = 0; j < M; j += nB) {
            src_ii = src_i;
            for (int ii = i; ii < i + nB; ii++) {
                for (int jj = j; jj < j + nB; jj++)
                    dst[N * jj + ii] = src[src_ii + jj];
                src_ii += M;
            }
        }
        src_i += nBM;
    }
}

void pad_matrix(double **M, int *r, int *c) {
    int temp_r;
    int temp_c;

    // THIS IF NEEDS MORE TESTING BECAUSE ADDING IT MAKES PADDED AND UNPADDED PERF THE SAME

    // if( ((*r) %BLOCK_SIZE_MMUL == 0 ) && ((*c)%BLOCK_SIZE_MMUL == 0)){
    //     return;
    // }

    if ((*r) % BLOCK_SIZE_MMUL != 0) {
        temp_r = (((*r) / BLOCK_SIZE_MMUL) + 1) * BLOCK_SIZE_MMUL;
    } else {
        temp_r = *r;
    }

    if ((*c) % BLOCK_SIZE_MMUL != 0) {
        temp_c = (((*c) / BLOCK_SIZE_MMUL) + 1) * BLOCK_SIZE_MMUL;
    } else {
        temp_c = *c;
    }

    double *new_Mt;

    *M = realloc(*M, double_size * (*c) * temp_r);
    // i need to pad the rows before and the cols after transposing
    memset(&(*M)[(*c) * (*r)], 0, double_size * (temp_r - (*r)) * (*c));

    new_Mt = malloc(double_size * temp_c * temp_r);
    transpose(*M, new_Mt, temp_r, *c);
    memset(&new_Mt[temp_r * (*c)], 0, double_size * (temp_c - (*c)) * temp_r);

    free(*M);
    *M = malloc(double_size * temp_c * temp_r);
    *c = temp_c;
    *r = temp_r;
    transpose(new_Mt, *M, temp_c, temp_r);

    free(new_Mt);


}

void unpad_matrix(double **M, int *r, int *c, int original_r, int original_c) {

    // lets suppose that are always row majour

    // i can remove the last useless rows
    *M = realloc(*M, (*c) * original_r * double_size);

    // i need to transpose and remove the rest
    double *new_Mt = malloc((*c) * original_r * double_size);
    transpose(*M, new_Mt, original_r, *c);

    // i need to resize the transoposed
    new_Mt = realloc(new_Mt, double_size * original_c * original_r);

    // ie need to transpose back
    free(*M);
    *M = malloc(double_size * original_c * original_r);
    transpose(new_Mt, *M, original_c, original_r);

    *r = original_r;
    *c = original_c;

    free(new_Mt);

}

void perf_m(Matrix A, Matrix B, Matrix C, int iterations) {

    myInt64 cost;
    double performance;

    myInt64 cycles;
    myInt64 start;

#ifdef CALIBRATE
    int num_runs = 1;
    while(num_runs < (1 << 7)) {
        start = start_tsc();
        for (int j = 0; j < num_runs; j++) {
            run_mmm_m(&A, &B, &C);
        }
        cycles = stop_tsc(start);

        if(cycles >= CYCLES_REQUIRED) break;

        num_runs *= 2;
    }
#endif

    start = start_tsc();
    for (int i = 0; i < iterations; i++) {
        run_mmm_m(&A, &B, &C);
    }

    cycles = stop_tsc(start);
    cost = ((myInt64)(2 * A.n_row * B.n_col * A.n_col)) * iterations;
    performance = (double) cost / (double) cycles;

    double seconds = ((double) (cycles)) / CPU_FREQ;

    printf("Cycles: %llu\tPerformance(f/c): %lf\tCost: %llu\tRuntime: %.4lfs\n", cycles, performance, cost, seconds);

}


void perf_v(double *A, double *B, double *C, int m, int n, int r, int iterations) {
    myInt64 cost;
    double performance;

    myInt64 cycles;
    myInt64 start;

    int original_m = m;
    int original_n = n;
    int original_r = r;


#ifdef CALIBRATE
    int num_runs = 1;
    while(num_runs < (1 << 7)) {
        start = start_tsc();
        for (int j = 0; j < num_runs; j++) {
            run_mmm_v(A, m, r, B, r, n, C, m, n);
        }
        cycles = stop_tsc(start);

        if(cycles >= CYCLES_REQUIRED) break;

        num_runs *= 2;
    }
#endif


    int temp_m = m;
    int temp_n = n;
    int temp_r = r;

    // PADDING DONE BEFORE TO NOT AFFECT THE COMPUTATION
    if (run_mmm_v == &matrix_mul_3) {
        pad_matrix(&A, &temp_m, &temp_r);
        pad_matrix(&B, &r, &temp_n);
        pad_matrix(&C, &m, &n);
    }

    int save_m = m;
    int save_n = n;
    int save_r = r;
    start = start_tsc();

    for (int i = 0; i < iterations; i++) {
        run_mmm_v(A, m, r, B, r, n, C, m, n);
    }

    cycles = stop_tsc(start);


    if (run_mmm_v == &matrix_mul_3) {
        temp_m = m;
        temp_n = n;
        temp_r = r;
        unpad_matrix(&A, &m, &temp_r, original_m, original_r);
        unpad_matrix(&B, &temp_n, &r, original_r, original_n);
        unpad_matrix(&C, &temp_m, &n, original_m, original_n);
    }


    cost = ((myInt64)(2 * save_m * save_n * save_r)) * iterations;
    performance = (double) cost / (double) cycles;

    double seconds = ((double) (cycles)) / CPU_FREQ;


    printf("Cycles: %llu\tPerformance(f/c): %lf\tCost: %llu\tRuntime: %.4lfs\n", cycles, performance, cost, seconds);

}

int main(int argc, char const *argv[]) {
    if (argc < 6) {
        printf("Usage:\n");
        printf("\t./build/run_mmm <id> <m> <n> <r> <num-iter>\n\n");

        printf("IDs:\n");
        printf("\t- 0: Baseline 1 mmm\n");
        printf("\t- 1: Baseline 2 mmm\n");
        printf("\t- 2: Opt 45 (best w/o blas) mmm\n");
        printf("\t- 3: Opt 46 (opt 45 padded) mmm\n");
        printf("\n");

        return 0;
    }


    int id = atoi(argv[1]);
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int r = atoi(argv[4]);
    int iter = atoi(argv[5]);

    fflush(stdout);
    Matrix A_m, B_m, C_m;
    double *A_v, *B_v, *C_v;

    matrix_allocation(&A_m, m, r);
    matrix_allocation(&B_m, r, n);
    matrix_allocation(&C_m, m, n);

    A_v = malloc(sizeof(double) * m * r);
    B_v = malloc(sizeof(double) * r * n);
    C_v = malloc(sizeof(double) * m * n);

    random_matrix_init(&A_m, 0, 1);
    random_matrix_init(&B_m, 0, 1);


    for (int i = 0; i < m * r; i++) {
        A_v[i] = rand_from(0, 1);
    }

    for (int i = 0; i < n * r; i++) {
        B_v[i] = rand_from(0, 1);
    }

    fflush(stdout);
    switch (id) {
        case 0:
            run_mmm_m = &matrix_mul_0;
            perf_m(A_m, B_m, C_m, iter);
            break;
        case 1:
            run_mmm_m = &matrix_mul_1;
            perf_m(A_m, B_m, C_m, iter);
            break;
        case 2:
            run_mmm_v = &matrix_mul_2;
            perf_v(A_v, B_v, C_v, m, n, r, iter);
            break;
        case 3:
            run_mmm_v = &matrix_mul_3;
            perf_v(A_v, B_v, C_v, m, n, r, iter);
            break;

        default:
            printf("Wrong id\n");
            break;
    }

    return 0;
}
