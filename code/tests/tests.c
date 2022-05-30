//#include "stdafx.h"  //Visual studio expects this line to be the first one, comment out if different compiler
//#include <windows.h> // Include if under windows

#include <tests/tests.h>


#define M_PERF M_TEST
#define N_PERF N_TEST
#define R_PERF RANK
#define TOLERANCE 0.000001


typedef struct {
    int n_row;
    int n_col;
} Sizes;

typedef struct {
    double perf;
    myInt64 cycles;
} PerfResults;

static void print_test_status(int return_value) {
    if (return_value == -1) {
        printf("\t\e[0;31mFAIL\e[0m");
    } else {
        printf("\t\e[32mSUCESS\e[0m");
    }
}

static void read_matrix_from_file(Matrix *M, FILE *f) {
    int n_row;
    int n_col;

    fscanf(f, "%d", &n_row);
    fscanf(f, "%d", &n_col);

    matrix_allocation(M, n_row, n_col);

    for (int i = 0; i < M->n_row * M->n_col; i++) {
        fscanf(f, "%lf", &(M->M[i]));
    }
}

static Sizes read_double_from_file(double **M, FILE *f) {
    int n_row;
    int n_col;
    Sizes s;

    fscanf(f, "%d", &n_row);
    fscanf(f, "%d", &n_col);

    s.n_col = n_col;
    s.n_row = n_row;

    *M = malloc(sizeof(double *) * n_row * n_col);
    double *t = *M;

    for (int i = 0; i < n_row * n_col; i++) {
        fscanf(f, "%lf", &(t[i]));
    }
    return s;
}

int test_matrix_mult_d(
        void (*mmuld)(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R, int R_n_row,
                      int R_n_col)) {
    double *A;
    double *B;
    double *R_Real;
    double *R_Computed;
    Sizes R_Real_size;
    Sizes A_size;
    Sizes B_size;
    int return_val = 0;

    FILE *f;

    f = fopen("A_mul.matrix", "r");
    if (f == NULL) {
        printf("Error opening file\n");
        return 0;
    }
    A_size = read_double_from_file(&A, f);
    fclose(f);

    f = fopen("B_mul.matrix", "r");
    B_size = read_double_from_file(&B, f);
    fclose(f);

    f = fopen("R_mul.matrix", "r");
    R_Real_size = read_double_from_file(&R_Real, f);
    fclose(f);

    R_Computed = malloc(sizeof(double *) * R_Real_size.n_col * R_Real_size.n_row);

    mmuld(A, A_size.n_row, A_size.n_col, B, B_size.n_row, B_size.n_col, R_Computed, R_Real_size.n_row,
          R_Real_size.n_col);

    for (int i = 0; i < R_Real_size.n_col * R_Real_size.n_row; i++) {
        if (fabs(R_Computed[i] - R_Real[i]) > TOLERANCE) {
            return_val = -1;
            break;
        }
    }

    free(A);
    free(B);
    free(R_Real);
    free(R_Computed);
    return return_val;
}

int test_matrix_ltrans_mult_d(
        void (*mmulltransd)(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R,
                            int R_n_row, int R_n_col)) {
    double *A;
    double *B;
    double *R_Real;
    double *R_Computed;
    Sizes R_Real_size;
    Sizes A_size;
    Sizes B_size;
    int return_val = 0;

    FILE *f;

    f = fopen("A_ltrans_mul.matrix", "r");
    if (f == NULL) {
        printf("Error opening file\n");
        return 0;
    }
    A_size = read_double_from_file(&A, f);
    fclose(f);

    f = fopen("B_ltrans_mul.matrix", "r");
    B_size = read_double_from_file(&B, f);
    fclose(f);

    f = fopen("R_ltrans_mul.matrix", "r");
    R_Real_size = read_double_from_file(&R_Real, f);
    fclose(f);

    R_Computed = malloc(sizeof(double *) * R_Real_size.n_col * R_Real_size.n_row);
    mmulltransd(A, A_size.n_row, A_size.n_col, B, B_size.n_row, B_size.n_col, R_Computed, R_Real_size.n_row,
                R_Real_size.n_col);

    for (int i = 0; i < R_Real_size.n_col * R_Real_size.n_row; i++) {
        if (fabs(R_Computed[i] - R_Real[i]) > TOLERANCE) {
            return_val = -1;
            break;
        }
    }

    free(A);
    free(B);
    free(R_Real);
    free(R_Computed);
    return return_val;
}

int test_matrix_rtrans_mult_d(
        void (*mmulrtransd)(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R,
                            int R_n_row, int R_n_col)) {
    double *A;
    double *B;
    double *R_Real;
    double *R_Computed;
    Sizes R_Real_size;
    Sizes A_size;
    Sizes B_size;
    int return_val = 0;

    FILE *f;

    f = fopen("A_rtrans_mul.matrix", "r");
    if (f == NULL) {
        printf("Error opening file\n");
        return 0;
    }
    A_size = read_double_from_file(&A, f);
    fclose(f);

    f = fopen("B_rtrans_mul.matrix", "r");
    B_size = read_double_from_file(&B, f);
    fclose(f);

    f = fopen("R_rtrans_mul.matrix", "r");
    R_Real_size = read_double_from_file(&R_Real, f);
    fclose(f);

    R_Computed = malloc(sizeof(double *) * R_Real_size.n_col * R_Real_size.n_row);
    mmulrtransd(A, A_size.n_row, A_size.n_col, B, B_size.n_row, B_size.n_col, R_Computed, R_Real_size.n_row,
                R_Real_size.n_col);

    for (int i = 0; i < R_Real_size.n_col * R_Real_size.n_row; i++) {
        if (fabs(R_Computed[i] - R_Real[i]) > TOLERANCE) {
            return_val = -1;
            break;
        }
    }

    free(A);
    free(B);
    free(R_Real);
    free(R_Computed);
    return return_val;
}

int test_nnm_d(double (*nnmd)(double *V, double *W, double *H, int m, int n, int r, int maxIteration, double epsilon)) {

    double resultBS1, resultBS2;
    srand(SEED);
    int m = M_PERF;
    int n = N_PERF;
    int r = R_PERF;

    double *V;
    double *W;
    double *H;

    int mr, rn, mn;

    double rand_max_r = 1 / (double) RAND_MAX;

    mr = m * r;
    rn = n * r;
    mn = m * n;

    V = malloc(sizeof(double *) * m * n);
    W = malloc(sizeof(double *) * m * r);
    H = malloc(sizeof(double *) * r * n);

    for (int i = 0; i < mn; i++) {
        V[i] = rand() * rand_max_r;
    }

    for (int i = 0; i < mr; i++) {
        W[i] = rand() * rand_max_r;
    }

    for (int i = 0; i < rn; i++) {
        H[i] = rand() * rand_max_r;
    }

    // Setup for BS1 for comparison:
    Matrices matrices;

    allocate_base_matrices(&matrices, m, n, r);

    for (int i = 0; i < matrices.W.n_col * matrices.W.n_row; i++) {
        matrices.W.M[i] = W[i];
    }

    for (int i = 0; i < matrices.H.n_col * matrices.H.n_row; i++) {
        matrices.H.M[i] = H[i];
    }

    for (int i = 0; i < matrices.V.n_col * matrices.V.n_row; i++) {
        matrices.V.M[i] = V[i];
    }

    // Run:

    resultBS1 = nnm_factorization_bs1(&matrices.V, &matrices.W,
                                      &matrices.H, MAX_ITERATIONS, EPSILON);

    resultBS2 = nnmd(V, W, H, m, n, r, MAX_ITERATIONS, EPSILON);
    if (isnan(resultBS2) || isinf(resultBS2)) { return -1; }
    if (fabs(resultBS1 - resultBS2) > TOLERANCE) {
        printf("Results: error_bs1=%lf, error_implementation=%lf, error=%lf\t", resultBS1, resultBS2,
               fabs(resultBS1 - resultBS2));
        return -1;
    }

    for (int i = 0; i < matrices.H.n_col * matrices.H.n_row; i++) {
        if (fabs(matrices.H.M[i] - H[i]) > TOLERANCE) {
            return -1;
        }
    }

    for (int i = 0; i < matrices.W.n_col * matrices.W.n_row; i++) {
        if (fabs(matrices.W.M[i] - W[i]) > TOLERANCE) {
            return -1;
        }
    }
    return 0;
}

int test_matrix_mult(void (*mmul)(Matrix *A, Matrix *B, Matrix *R)) {
    Matrix A;
    Matrix B;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("A_mul.matrix", "r");
    if (f == NULL) {
        printf("Error opening file\n");
        return 0;
    }
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("B_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("R_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    mmul(&A, &B, &R_Computed);

    for (int i = 0; i < R_Computed.n_col * R_Computed.n_row; i++) {
        if (fabs(R_Real.M[i] - R_Computed.M[i]) > TOLERANCE) {
            printf("ERROR: %lf\t", fabs(R_Real.M[i] - R_Computed.M[i]));
            matrix_deallocation(&A);
            matrix_deallocation(&B);
            matrix_deallocation(&R_Real);
            matrix_deallocation(&R_Computed);
            return -1;
        }
    }

    matrix_deallocation(&A);
    matrix_deallocation(&B);
    matrix_deallocation(&R_Real);

    return 0;

}

int test_matrix_ltrans_mult(void (*mmulltrans)(Matrix *A, Matrix *B, Matrix *R)) {
    Matrix A;
    Matrix B;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("A_ltrans_mul.matrix", "r");
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("B_ltrans_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("R_ltrans_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    mmulltrans(&A, &B, &R_Computed);

    for (int i = 0; i < R_Computed.n_col * R_Computed.n_row; i++) {
        if (fabs(R_Real.M[i] - R_Computed.M[i]) > TOLERANCE) {
            matrix_deallocation(&A);
            matrix_deallocation(&B);
            matrix_deallocation(&R_Real);
            return -1;
        }
    }

    matrix_deallocation(&A);
    matrix_deallocation(&B);
    matrix_deallocation(&R_Real);

    return 0;

}

int test_matrix_rtrans_mult(void (*mmulrtrans)(Matrix *A, Matrix *B, Matrix *R)) {
    Matrix A;
    Matrix B;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("A_rtrans_mul.matrix", "r");
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("B_rtrans_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("R_rtrans_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    mmulrtrans(&A, &B, &R_Computed);

    for (int i = 0; i < R_Computed.n_col * R_Computed.n_row; i++) {
        if (fabs(R_Real.M[i] - R_Computed.M[i]) > TOLERANCE) {
            matrix_deallocation(&A);
            matrix_deallocation(&B);
            matrix_deallocation(&R_Real);
            return -1;
        }
    }

    matrix_deallocation(&A);
    matrix_deallocation(&B);
    matrix_deallocation(&R_Real);

    return 0;
}

int test_nnm(double (*nnm)(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon)) {

    double resultBS1, resultBS2;
    srand(SEED);
    int m = 400;
    int n = 400;
    int r = 3;
    int min = 0;
    int max = 100;

    Matrices matrices;
    Matrix W_temp;
    Matrix H_temp;

    allocate_base_matrices(&matrices, m, n, r);

    random_matrix_init(&matrices.V, min, max);
    random_matrix_init(&matrices.W, min, max);

    random_matrix_init(&matrices.H, min, max);

    // copy the matrices
    matrix_allocation(&W_temp, matrices.W.n_row, matrices.W.n_col);
    matrix_allocation(&H_temp, matrices.H.n_row, matrices.H.n_col);

    for (int i = 0; i < matrices.W.n_col * matrices.W.n_row; i++) {
        W_temp.M[i] = matrices.W.M[i];
    }

    for (int i = 0; i < matrices.H.n_col * matrices.H.n_row; i++) {
        H_temp.M[i] = matrices.H.M[i];
    }

    resultBS1 = nnm_factorization_bs1(&matrices.V, &matrices.W,
                                      &matrices.H, MAX_ITERATIONS, EPSILON);

    resultBS2 = nnm(&matrices.V, &W_temp,
                    &H_temp, MAX_ITERATIONS, EPSILON);
    if (fabs(resultBS1 - resultBS2) > TOLERANCE) {
        printf("Results: error_bs1=%lf, error_implementation=%lf\t", resultBS1, resultBS2);
        return -1;
    }

    for (int i = 0; i < matrices.H.n_col * matrices.H.n_row; i++) {
        if (fabs(H_temp.M[i] - matrices.H.M[i]) > TOLERANCE) {
            printf("H_bs1[%d][%d] - H_implementation[%d][%d] diff by %lf\t", i, i / H_temp.n_col, i, i / H_temp.n_col,
                   fabs(H_temp.M[i] - matrices.H.M[i]));
            return -1;
        }
    }

    for (int i = 0; i < matrices.W.n_col * matrices.W.n_row; i++) {
        if (fabs(W_temp.M[i] - matrices.W.M[i]) > TOLERANCE) {
            printf("W_bs1[%d][%d] - W_implementation[%d][%d] diff by %lf\t", i, i / W_temp.n_col, i, i / W_temp.n_col,
                   fabs(W_temp.M[i] - matrices.W.M[i]));
            return -1;
        }
    }
    return 0;

}

PerfResults performance_analysis_matrix_mult(void (*mmul)(Matrix *A, Matrix *B, Matrix *R)) {
    Matrix V;
    Matrix W, H;

    myInt64 performance = 0;
    myInt64 cost = 0;

    for (int i = 0; i < NUM_RUNS; i++) {

        // REMINDER This is the test for the matrix mult and not for the nnmf so the matrices can have different sizes

        matrix_allocation(&V, M_PERF, M_PERF);
        matrix_allocation(&W, M_PERF, M_PERF);
        matrix_allocation(&H, M_PERF, M_PERF);


#ifdef __x86_64__
        myInt64 cycles;
        myInt64 start;

#ifdef CALIBRATE
        while(num_runs < (1 << 14)) {
            start = start_tsc();
            for (int j = 0; j < num_runs; j++) {

                random_matrix_init(&W, 0, 1);
                random_matrix_init(&H, 0, 1);
                random_matrix_init(&V, 0, 1);

                mmul(&V, &W, &H);
            }
            cycles = stop_tsc(start);

            if(cycles >= CYCLES_REQUIRED) break;

            num_runs *= 2;
        }
#endif


        random_matrix_init(&W, 0, 1);
        random_matrix_init(&H, 0, 1);
        random_matrix_init(&V, 0, 1);

        start = start_tsc();
        mmul(&V, &W, &H);

        cycles = stop_tsc(start);
        performance += cycles;
        cost += matrix_mul_cost(M_PERF, M_PERF, M_PERF);

#endif
    }
    PerfResults results = {.cycles = performance / NUM_RUNS, .perf = ((double) cost / performance)};
    return results;
}

PerfResults performance_analysis_matrix_mult_d(
        void (*mmuld)(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R, int R_n_row,
                      int R_n_col)) {
    double *V;
    double *W;
    double *H;
    double rand_max_r = 1 / (double) RAND_MAX;

    myInt64 performance = 0;
    myInt64 cost = 0;
    int num_runs = NUM_RUNS;
    int mr = M_PERF * M_PERF;
    int rn = M_PERF * M_PERF;
    int mn = M_PERF * M_PERF;

    for (int i = 0; i < num_runs; i++) {

        V = malloc(sizeof(double *) * mr);
        W = malloc(sizeof(double *) * rn);
        H = malloc(sizeof(double *) * mn);

#ifdef __x86_64__
        myInt64 cycles;
        myInt64 start;

#ifdef CALIBRATE
        while(num_runs < (1 << 14)) {
            start = start_tsc();
            for (int j = 0; j < num_runs; j++) {

                for (int i = 0; i < mr; i++)
                    V[i] = rand() * rand_max_r;
                for (int i = 0; i < nr; i++)
                    W[i] = rand() * rand_max_r;
                for (int i = 0; i < mn; i++)
                    H[i] = rand() * rand_max_r;

                mmuld(V, M_PERF, M_PERF, W, M_PERF, M_PERF, H, M_PERF, M_PERF);
            }
            cycles = stop_tsc(start);

            if(cycles >= CYCLES_REQUIRED) break;

            num_runs *= 2;
        }
#endif


        for (int i = 0; i < mr; i++)
            V[i] = rand() * rand_max_r;
        for (int i = 0; i < rn; i++)
            W[i] = rand() * rand_max_r;
        for (int i = 0; i < mn; i++)
            H[i] = rand() * rand_max_r;

        start = start_tsc();
        mmuld(V, M_PERF, M_PERF, W, M_PERF, M_PERF, H, M_PERF, M_PERF);

        cycles = stop_tsc(start);
        performance += cycles;
        cost += matrix_mul_cost(M_PERF, M_PERF, M_PERF);

#endif
    }
    PerfResults results = {.cycles = performance / NUM_RUNS, .perf = ((double) cost / performance)};
    return results;
}

PerfResults performance_analysis_nnm(double (*nnm)(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon)) {
    Matrix V;
    Matrix W, H;

    myInt64 performance = 0;
    myInt64 cost = 0;

    matrix_allocation(&V, M_PERF, N_PERF);
    matrix_allocation(&W, M_PERF, R_PERF);
    matrix_allocation(&H, R_PERF, N_PERF);

    for (int i = 0; i < NUM_RUNS; i++) {


#ifdef __x86_64__
        myInt64 cycles;
        myInt64 start;

#ifdef CALIBRATE
        while(num_runs < (1 << 14)) {
            start = start_tsc();
            for (int j = 0; j < num_runs; j++) {

                random_matrix_init(&W, 0, 1);
                random_matrix_init(&H, 0, 1);
                random_matrix_init(&V, 0, 1);

                nnm(&V, &W, &H, maxIterations, epsilon);
            }
            cycles = stop_tsc(start);

            if(cycles >= CYCLES_REQUIRED) break;

            num_runs *= 2;
        }
#endif

        start = start_tsc();
        random_matrix_init(&W, 0, 1);
        random_matrix_init(&H, 0, 1);
        random_matrix_init(&V, 0, 1);

        nnm(&V, &W, &H, MAX_ITERATIONS, EPSILON);

        cycles = stop_tsc(start);
        performance += cycles;
        cost += nnm_cost(M_PERF, N_PERF, M_PERF, R_PERF, R_PERF, N_PERF, MAX_ITERATIONS) + matrix_rand_init_cost(M_PERF, N_PERF) + matrix_rand_init_cost(M_PERF, R_PERF) + matrix_rand_init_cost(R_PERF, N_PERF);


#endif
    }
    PerfResults results = {.cycles = performance / NUM_RUNS, .perf = ((double) cost / performance)};
    return results;
}

PerfResults performance_analysis_nnm_d(
        double (*nnmd)(double *V, double *W, double *H, int m, int n, int r, int maxIteration, double epsilon)) {
    double *V;
    double *W;
    double *H;

    int mr = M_PERF * R_PERF;
    int rn = R_PERF * N_PERF;
    int mn = M_PERF * N_PERF;
    double rand_max_r = 1 / (double) RAND_MAX;

    myInt64 performance = 0;
    myInt64 cost = 0;

    V = malloc(sizeof(double *) * mr);
    W = malloc(sizeof(double *) * rn);
    H = malloc(sizeof(double *) * mn);

    for (int i = 0; i < NUM_RUNS; i++) {

#ifdef __x86_64__
        myInt64 cycles;
        myInt64 start;

#ifdef CALIBRATE
        while(num_runs < (1 << 14)) {
            start = start_tsc();
            for (int j = 0; j < num_runs; j++) {

                for (int i = 0; i < mr; i++)
                    V[i] = rand() * rand_max_r;
                for (int i = 0; i < nr; i++)
                    W[i] = rand() * rand_max_r;
                for (int i = 0; i < mn; i++)
                    H[i] = rand() * rand_max_r;

                nnmd(V, W, H, M_PERF, N_PERF, R_PERF, maxIterations, epsilon);
            }
            cycles = stop_tsc(start);

            if(cycles >= CYCLES_REQUIRED) break;

            num_runs *= 2;
        }
#endif

        start = start_tsc();
        for (int i = 0; i < mr; i++)
            V[i] = rand() * rand_max_r;
        for (int i = 0; i < rn; i++)
            W[i] = rand() * rand_max_r;
        for (int i = 0; i < mn; i++)
            H[i] = rand() * rand_max_r;

        nnmd(V, W, H, M_PERF, N_PERF, R_PERF, MAX_ITERATIONS, EPSILON);

        cycles = stop_tsc(start);
        performance += cycles;
        cost += nnm_cost_2(M_PERF, N_PERF, M_PERF, R_PERF, R_PERF, N_PERF, MAX_ITERATIONS) + matrix_rand_init_cost(M_PERF, N_PERF) + matrix_rand_init_cost(M_PERF, R_PERF) + matrix_rand_init_cost(R_PERF, N_PERF);


#endif
    }
    PerfResults results = {.cycles = performance / NUM_RUNS, .perf = ((double) cost / performance)};
    return results;
}

void run_tests(
        int n_mmul,
        void (*mmul[n_mmul])(Matrix *A, Matrix *B, Matrix *R),
        int n_mulltrans,
        void (*mmulltrans[n_mulltrans])(Matrix *A, Matrix *B, Matrix *R),
        int n_mmulrtrans,
        void (*mmulrtrans[n_mmulrtrans])(Matrix *A, Matrix *B, Matrix *R),
        int n_nnm,
        double (*nnm[n_nnm])(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon)
) {
    printf("################ Starting baseline tests ################\n\n");
    printf("Execution\t\t\t\t\tTest\tPerformance\n");

    int result;
    int sum_results = 0;
    PerfResults perf_res;


    for (int i = 0; i < n_mmul; i++) {
        printf("Matrix mult implementation %i:\t\t", i);
        result = test_matrix_mult(mmul[i]);
        print_test_status(result);
        sum_results += result;
        perf_res = performance_analysis_matrix_mult(mmul[i]);
        printf("\tcycles: %11llu\t", perf_res.cycles);
        printf("f/c: %3.2lf\n", perf_res.perf);
    }
    printf("\n");

    for (int i = 0; i < n_mulltrans; i++) {
        printf("Matrix ltrans mult implementation %i:\t", i);
        result = test_matrix_ltrans_mult(mmulltrans[i]);
        print_test_status(result);
        sum_results += result;
        perf_res = performance_analysis_matrix_mult(mmulltrans[i]);
        printf("\tcycles: %11llu\t", perf_res.cycles);
        printf("f/c: %3.2lf\n", perf_res.perf);
    }
    printf("\n");

    for (int i = 0; i < n_mmulrtrans; i++) {
        printf("Matrix rtrans mult implementation %i:\t", i);
        result = test_matrix_rtrans_mult(mmulrtrans[i]);
        print_test_status(result);
        sum_results += result;
        perf_res = performance_analysis_matrix_mult(mmulrtrans[i]);
        printf("\tcycles: %11llu\t", perf_res.cycles);
        printf("f/c: %3.2lf\n", perf_res.perf);
    }
    printf("\n");

    for (int i = 0; i < n_nnm; i++) {
        printf("NNM implementation %i:\t\t\t", i);
        result = test_nnm(nnm[i]);
        print_test_status(result);
        sum_results += result;
        perf_res = performance_analysis_nnm(nnm[i]);
        printf("\tcycles: %11llu\t", perf_res.cycles);
        printf("f/c: %3.2lf\n", perf_res.perf);
    }

    if (sum_results == 0) {
        printf("\nTest completed. All test \e[32mPASSED\e[0m\n");
    } else {
        printf("\nTest completed. Numer of test \e[0;31mFAILED\e[0m: %d\n", sum_results * -1);
    }
}

void run_tests_d(
        int n_mmul_opt,
        void (*mmuld[n_mmul_opt])(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R,
                                  int R_n_row, int R_n_col),
        int n_mmulrtrans_opt,
        void (*mmulrtransd[n_mmulrtrans_opt])(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col,
                                              double *R, int R_n_row, int R_n_col),
        int n_mulltrans_opt,
        void (*mmulltransd[n_mulltrans_opt])(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col,
                                             double *R, int R_n_row, int R_n_col),
        int n_nnm_opt,
        double (*nnmd[n_nnm_opt])(double *V, double *W, double *H, int m, int n, int r, int maxIteration,
                                  double epsilon)
) {
    printf("################ Starting optimization tests ################\n\n");
    printf("Execution\t\t\t\t\tTest\tPerformance\n");

    int result;
    int sum_results = 0;
    PerfResults perf_res;

    for (int i = 0; i < n_mmul_opt; i++) {
        printf("Matrix mult optimization %i:\t\t", i);
        result = test_matrix_mult_d(mmuld[i]);
        print_test_status(result);
        sum_results += result;
        perf_res = performance_analysis_matrix_mult_d(mmuld[i]);
        printf("\tcycles: %11llu\t", perf_res.cycles);
        printf("f/c: %.2lf\n", perf_res.perf);
    }
    printf("\n");

    for (int i = 0; i < n_mulltrans_opt; i++) {
        printf("Matrix ltrans mult optimization %i:\t", i);
        result = test_matrix_ltrans_mult_d(mmulltransd[i]);
        print_test_status(result);
        sum_results += result;
        perf_res = performance_analysis_matrix_mult_d(mmulltransd[i]);
        printf("\tcycles: %11llu\t", perf_res.cycles);
        printf("f/c: %.2lf\n", perf_res.perf);
    }
    printf("\n");

    for (int i = 0; i < n_mmulrtrans_opt; i++) {
        printf("Matrix rtrans mult optimization %i:\t", i);
        result = test_matrix_rtrans_mult_d(mmulrtransd[i]);
        print_test_status(result);
        sum_results += result;
        perf_res = performance_analysis_matrix_mult_d(mmulrtransd[i]);
        printf("\tcycles: %11llu\t", perf_res.cycles);
        printf("f/c: %.2lf\n", perf_res.perf);
    }
    printf("\n");

    for (int i = 0; i < n_nnm_opt; i++) {
        printf("NNM optimization %i:\t\t\t", i);
        result = test_nnm_d(nnmd[i]);
        print_test_status(result);
        sum_results += result;
        perf_res = performance_analysis_nnm_d(nnmd[i]);
        printf("\tcycles: %11llu\t", perf_res.cycles);
        printf("f/c: %3.2lf\n", perf_res.perf);
    }

    if (sum_results == 0) {
        printf("\nTest completed. All test \e[32mPASSED\e[0m\n");
    } else {
        printf("\nTest completed. Numer of test \e[0;31mFAILED\e[0m: %d\n", sum_results * -1);
    }
}


int main(int argc, char const *argv[]) {
    char *default_path = "/home/asl/asl-2022/code/tests/inputs";
    if (argc == 1) {
        printf("No path specified, using default: %s \n", default_path);
        chdir(default_path);
    } else {
        chdir(argv[1]);
    }

    // TODO: Please add following the number of functions you would like to test:
    int n_mmulrtrans = 0; //BACK 2
    int n_mmulltrans = 0; //BACK 2
    int n_mmul = 0; //BACK 2
    int n_nnm = 0; //BACK 2

    void (*mmulrtrans[n_mmulrtrans])(Matrix *A, Matrix *B, Matrix *R);
    void (*mmulltrans[n_mmulltrans])(Matrix *A, Matrix *B, Matrix *R);
    void (*mmul[n_mmul])(Matrix *A, Matrix *B, Matrix *R);
    double (*nnm[n_nnm])(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon);

    int n_mmulrtrans_opt = 0; //BACK 19
    int n_mmulltrans_opt = 0;
    int n_mmul_opt = 0; //BACK 23
    int n_nnm_opt = 2; //BACK 23

    void (*mmulrtransd[n_mmulrtrans_opt])(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col,
                                          double *R, int R_n_row, int R_n_col);
    void (*mmulltransd[n_mmulltrans_opt])(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col,
                                          double *R, int R_n_row, int R_n_col);
    void (*mmuld[n_mmul_opt])(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R,
                              int R_n_row, int R_n_col);
    double (*nnmd[n_nnm_opt])(double *V, double *W, double *H, int m, int n, int r, int maxIteration, double epsilon);

    // START TODO:
    // Now add the functions you would like to test from the impleemntations:
    /*mmul[0] = matrix_mul_bs1;
    mmul[1] = matrix_mul_bs2;

    mmulltrans[0] = matrix_ltrans_mul_bs1;
    mmulltrans[1] = matrix_ltrans_mul_bs2;

    mmulrtrans[0] = matrix_rtrans_mul_bs1;
    mmulrtrans[1] = matrix_rtrans_mul_bs2;

    nnm[0] = nnm_factorization_bs1;
    nnm[1] = nnm_factorization_bs2;

    run_tests(n_mmul, mmul, n_mmulltrans, mmulltrans, n_mmulrtrans, mmulrtrans, n_nnm, nnm);

    mmuld[0] = matrix_mul_opt0;
    mmuld[1] = matrix_mul_opt1;
    mmuld[2] = matrix_mul_aopt1;
    mmuld[3] = matrix_mul_aopt2;
    mmuld[4] = matrix_mul_opt2;
    mmuld[5] = matrix_mul_opt3;
    mmuld[6] = matrix_mul_opt21;
    mmuld[7] = matrix_mul_opt22;
    mmuld[8] = matrix_mul_opt23;
    mmuld[9] = matrix_mul_opt24;
    mmuld[10] = matrix_mul_opt31;
    mmuld[11] = matrix_mul_opt32;
    mmuld[12] = matrix_mul_opt33;
    mmuld[13] = matrix_mul_opt34;
    mmuld[14] = matrix_mul_opt35;
    mmuld[15] = matrix_mul_opt36;
    mmuld[16] = matrix_mul_opt41;
    mmuld[17] = matrix_mul_opt42;
    mmuld[18] = matrix_mul_opt43;
    mmuld[19] = matrix_mul_opt44;
    mmuld[20] = matrix_mul_opt45;
    mmuld[21] = matrix_mul_opt46_padding;
    mmuld[22] = matrix_mul_opt47_padding;

    mmulrtransd[0] = matrix_rtrans_mul_opt0;
    mmulrtransd[1] = matrix_rtrans_mul_opt1;
    mmulrtransd[2] = matrix_rtrans_mul_aopt1;
    mmulrtransd[3] = matrix_rtrans_mul_aopt2;
    mmulrtransd[4] = matrix_rtrans_mul_opt2;
    mmulrtransd[5] = matrix_rtrans_mul_opt3;
    mmulrtransd[6] = matrix_rtrans_mul_opt21;
    mmulrtransd[7] = matrix_rtrans_mul_opt22;
    mmulrtransd[8] = matrix_rtrans_mul_opt23;
    mmulrtransd[9] = matrix_rtrans_mul_opt24;
    mmulrtransd[10] = matrix_rtrans_mul_opt31;
    mmulrtransd[11] = matrix_rtrans_mul_opt32;
    mmulrtransd[12] = matrix_rtrans_mul_opt33;
    mmulrtransd[13] = matrix_rtrans_mul_opt34;
    mmulrtransd[14] = matrix_rtrans_mul_opt35;
    mmulrtransd[15] = matrix_rtrans_mul_opt36;
    mmulrtransd[16] = matrix_rtrans_mul_opt41;
    mmulrtransd[17] = matrix_rtrans_mul_opt42;
    mmulrtransd[18] = matrix_rtrans_mul_opt43;

    nnmd[0] = nnm_factorization_opt0;
    nnmd[1] = nnm_factorization_opt1;
    nnmd[2] = nnm_factorization_aopt1;
    nnmd[3] = nnm_factorization_aopt2;
    nnmd[4] = nnm_factorization_opt2;
    nnmd[5] = nnm_factorization_opt3;
    nnmd[6] = nnm_factorization_opt21;
    nnmd[7] = nnm_factorization_opt22;
    nnmd[8] = nnm_factorization_opt23;
    nnmd[9] = nnm_factorization_opt24;
    nnmd[10] = nnm_factorization_opt31;
    nnmd[11] = nnm_factorization_opt32;
    nnmd[12] = nnm_factorization_opt33;
    nnmd[13] = nnm_factorization_opt34;
    nnmd[14] = nnm_factorization_opt35;
    nnmd[15] = nnm_factorization_opt36;
    nnmd[16] = nnm_factorization_opt41;
    nnmd[17] = nnm_factorization_opt42;
    nnmd[18] = nnm_factorization_opt43;
    nnmd[19] = nnm_factorization_opt44;
    nnmd[20] = nnm_factorization_opt45;
    nnmd[21] = nnm_factorization_opt46;
    nnmd[22] = nnm_factorization_opt47;*/

    nnmd[0] = nnm_factorization_opt47;
    nnmd[1] = nnm_factorization_opt37;

    // END TODO

    run_tests_d(n_mmul_opt, mmuld, n_mmulrtrans_opt, mmulrtransd, n_mmulltrans_opt, mmulltransd, n_nnm_opt, nnmd);
    return 0;
}
