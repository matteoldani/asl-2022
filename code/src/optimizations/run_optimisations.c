#include <optimizations/optimizations_0.h>
#include <optimizations/optimizations_1.h>
#include <optimizations/alg_opt_1.h>
#include <optimizations/alg_opt_2.h>
#include <optimizations/optimizations_2.h>
#include <optimizations/optimizations_3.h>
#include <asl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef double (*fact_function) (double *, double*, double*, int, int, int, int, double);
static unsigned int double_size = sizeof(double);

/**
 * @brief prints the matrix
 * @param matrix    the matrix to be printed
 * @param n_row     number of rows in the matrix
 * @param n_col     number of columns in the martix
 */
void print_matrix_opt(double* matrix, int n_row, int n_col) {

    printf("Printing a matrix with %d rows and %d cols\n\n", n_row, n_col);
    for (int row = 0; row < n_row; row++) {
        for (int col = 0; col < n_col; col++) {
            fprintf(stdout, "%.2lf\t", matrix[row * n_col + col]);
        }
        fprintf(stdout, "\n\n");
    }
    fprintf(stdout, "\n\n");
}

/**
 * @brief   allocates the correspondin matrix
 *          and fill the matrix with the values
 *          read from the file
 *
 * @param matrix    the matrix that will be filled
 * @param m         is the number of rows in the matrix to be factorized
 * @param n         is the nunmber of columns in the matrix to be factorized
 * @param r         is the factorization parameter
 * @param file      is the file to read the matrix from
 */
void allocate_from_file_opt(double* matrix, int* m, int* n, int* r, FILE* file) {

    printf("Reading matrix information: \n");

    fscanf(file, "%d", r);
    printf("\tr: %d\n", *r);
    fscanf(file, "%d", m);
    printf("\tn_row: %d\n", *m);
    fscanf(file, "%d", n);
    printf("\tn_col: %d\n", *n);

    int mn = (*m) * (*n);

    matrix = malloc(sizeof(double) * mn);

    for (int i = 0; i < mn; i++)
        fscanf(file, "%lf", matrix + i);
}

int main(int argc, char const* argv[]) {

    double *V = NULL;      //m x n
    double *W, *H;  //m x r and r x n
    fact_function run_factorization = NULL;

    int m, n, r, opt_num = -1;
    m = 1000;
    n = 1000;
    r = 12;

    int mn, nr, mr;

    double rand_max_r = 1 / (double)RAND_MAX;

    if (argc != 3 && argc != 5) {
        printf("This program can be run in tow different modes:\n");
        printf("\t./ <#-opt> <m> <n> <r>\n");
        printf("\t./ <#-opt> <file-path>\n");

        // add here the optimisation descriptions
        printf("List of optimisations available:\n");
        printf("\t0 - Optimisation 0\n");
        printf("\t1 - Optimisation 1\n");
        printf("\t2 - Algoritmh opt 1\n");
        printf("\t3 - Algoritmh opt 2\n");
        printf("\t4 - Optimization 2\n");
        printf("\t5 - Optimization 3\n");
        return -1;
    }
    FILE* fp = NULL;
    if (argc == 5) {
        opt_num = atoi(argv[1]);
        m = atoi(argv[2]); 
        n = atoi(argv[3]);
        r = atoi(argv[4]);

        srand(SEED);

        mn = m * n;
        V = malloc(double_size * mn);
        for (int i = 0; i < mn; i++)
            V[i] = rand() * rand_max_r;
    }
    
    if (argc == 2) {
        opt_num = atoi(argv[1]);
        fp = fopen(argv[2], "r");
        if (fp == NULL) {
            printf("File not found, exiting\n");
            return -1;
        }

        allocate_from_file_opt(V, &m, &n, &r, fp);
    }

    mr = m * r;
    nr = n * r;
    W = malloc(double_size * mr);
    H = malloc(double_size * nr);

    for (int i = 0; i < mr; i++)
        W[i] = rand() * rand_max_r;
    for (int i = 0; i < nr; i++)
        H[i] = rand() * rand_max_r;
   

    switch (opt_num)
    {
    case 0:
        run_factorization = &nnm_factorization_opt0;
        break;
    case 1:
        run_factorization = &nnm_factorization_opt1;
        break;
    case 2:
        run_factorization = &nnm_factorization_aopt1;
        break;
    case 3:
        run_factorization = &nnm_factorization_aopt2;
        break;
    case 4:
        run_factorization = &nnm_factorization_opt2;
        break;
    case 5:
        run_factorization = &nnm_factorization_opt3;
        break;
    default:
        printf("Invalid opt number. Quitting\n");
        return -1;
    }

    double err = run_factorization(V, W, H, m, n, r, MAX_ITERATIONS, EPSILON);
    printf("Error: %lf\n", err);

    free(V);
    //printf("Free o V done\n");
    fflush(stdout);
    free(W);
    //printf("Free o W done\n");
    fflush(stdout);
    free(H);
    //printf("Free o H done\n");
    fflush(stdout);

    return 0;
}