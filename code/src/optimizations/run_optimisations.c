#include <optimizations/optimizations_0.h>
#include <optimizations/optimizations_1.h>
#include <optimizations/alg_opt_1.h>
#include <optimizations/alg_opt_2.h>
#include <optimizations/optimizations_2.h>
#include <optimizations/optimizations_3.h>
#include <optimizations/optimizations_21.h>
#include <optimizations/optimizations_22.h>
#include <optimizations/optimizations_23.h>
#include <optimizations/optimizations_24.h>
#include <optimizations/optimizations_31.h>
#include <optimizations/optimizations_32.h>
#include <optimizations/optimizations_33.h>
#include <optimizations/optimizations_34.h>
#include <optimizations/optimizations_35.h>
#include <optimizations/optimizations_36.h>
#include <optimizations/optimizations_41.h>
#include <optimizations/optimizations_42.h>
#include <optimizations/optimizations_43.h>
#include <optimizations/optimizations_44.h>
#include <optimizations/optimizations_45.h>
#include <optimizations/optimizations_46.h>
#include <optimizations/optimizations_47.h>
#include <optimizations/optimizations_37.h>
#include <optimizations/optimizations_51.h>
#include <optimizations/optimizations_53.h>
#include <optimizations/optimizations_54.h>
#include <optimizations/optimizations_utils.h>
#include <asl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef double (*fact_function)(double *, double *, double *, int, int, int, int, double);

static unsigned int double_size = sizeof(double);

/**
 * @brief prints the matrix
 * @param matrix    the matrix to be printed
 * @param n_row     number of rows in the matrix
 * @param n_col     number of columns in the martix
 */
void print_matrix_opt(double *matrix, int n_row, int n_col) {

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
void allocate_from_file_opt(double *matrix, int *m, int *n, int *r, FILE *file) {

    printf("Reading matrix information: \n");

    fscanf(file, "%d", r);
    printf("\tr: %d\n", *r);
    fscanf(file, "%d", m);
    printf("\tn_row: %d\n", *m);
    fscanf(file, "%d", n);
    printf("\tn_col: %d\n", *n);

    int mn = (*m) * (*n);

    matrix = aligned_alloc(32, sizeof(double) * mn);

    for (int i = 0; i < mn; i++)
        fscanf(file, "%lf", matrix + i);
}

int main(int argc, char const *argv[]) {

    double *V = NULL;      //m x n
    double *W, *H;  //m x r and r x n
    fact_function run_factorization = NULL;

    int m, n, r, opt_num = -1;
    m = 1000;
    n = 1000;
    r = 12;

    int mn, nr, mr;

    double rand_max_r = 1 / (double) RAND_MAX;

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
        printf("\t6. Optimisation 21\n");
        printf("\t7. Optimisation 22\n");
        printf("\t8. Optimisation 23\n");
        printf("\t9. Optimisation 24\n");
        printf("\t10. Optimisation 31\n");
        printf("\t11. Optimisation 32\n");
        printf("\t12. Optimisation 33\n");
        printf("\t13. Optimisation 34\n");
        printf("\t14. Optimisation 35\n");
        printf("\t15. Optimisation 36\n");
        printf("\t16. Optimisation 41\n");
        printf("\t17. Optimisation 42\n");
        printf("\t18. Optimisation 43\n");
        printf("\t19. Optimisation 44\n");
        printf("\t20. Optimisation 45\n");
        printf("\t21. Optimisation 46\n");
        printf("\t22. Optimisation 47\n");
        printf("\t23. Optimisation 37\n");
        printf("\t24. Optimisation 51\n");
        printf("\t25. Optimisation 53\n");
        printf("\t26. Optimisation 54\n");


        return -1;
    }
    FILE *fp = NULL;
    if (argc == 5) {
        opt_num = atoi(argv[1]);
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        r = atoi(argv[4]);

        srand(SEED);

        mn = m * n;
        V = aligned_alloc(32, double_size * mn);
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
    W = aligned_alloc(32, double_size * mr);
    H = aligned_alloc(32, double_size * nr);

    for (int i = 0; i < mr; i++)
        W[i] = rand() * rand_max_r;
    for (int i = 0; i < nr; i++)
        H[i] = rand() * rand_max_r;


    switch (opt_num) {
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
        case 6:
            run_factorization = &nnm_factorization_opt21;
            break;
        case 7:
            run_factorization = &nnm_factorization_opt22;
            break;
        case 8:
            run_factorization = &nnm_factorization_opt23;
            break;
        case 9:
            run_factorization = &nnm_factorization_opt24;
            break;
        case 10:
            run_factorization = &nnm_factorization_opt31;
            break;
        case 11:
            run_factorization = &nnm_factorization_opt32;
            break;
        case 12:
            run_factorization = &nnm_factorization_opt33;
            break;
        case 13:
            run_factorization = &nnm_factorization_opt34;
            break;
        case 14:
            run_factorization = &nnm_factorization_opt35;
            break;
        case 15:
            run_factorization = &nnm_factorization_opt36;
            break;
        case 16:
            run_factorization = &nnm_factorization_opt41;
            break;
        case 17:
            run_factorization = &nnm_factorization_opt42;
            break;
        case 18:
            run_factorization = &nnm_factorization_opt43;
            break;
        case 19:
            run_factorization = &nnm_factorization_opt44;
            break;
        case 20:
            run_factorization = &nnm_factorization_opt45;
            break;
        case 21:
            run_factorization = &nnm_factorization_opt46;
            break;
        case 22:
            run_factorization = &nnm_factorization_opt47;
            break;
        case 23:
            run_factorization = &nnm_factorization_opt37;
            break;
        case 24:
            run_factorization = &nnm_factorization_opt51;
            break;
        case 25:
            run_factorization = &nnm_factorization_opt53;
            break;
        case 26:
            run_factorization = &nnm_factorization_opt54;
            break;
        default:
            printf("Invalid opt number. Quitting\n");
            return -1;
    }

    double err = run_factorization(V, W, H, m, n, r, MAX_ITERATIONS, EPSILON);
    printf("Error: %lf\n", err);

    free(V);
    free(W);
    free(H);
    fflush(stdout);

    return 0;
}