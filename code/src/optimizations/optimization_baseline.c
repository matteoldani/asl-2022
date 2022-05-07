#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

unsigned int double_size = sizeof(double);

/**
 * @brief compute the multiplication of A and B
 * @param A         is the first factor
 * @param A_n_row   is the number of rows in matrix A
 * @param A_n_col   is the number of columns in matrix A
 * @param B         is the other factor of the multiplication
 * @param B_n_row   is the number of rows in matrix B
 * @param B_n_col   is the number of columns in matrix B
 * @param R         is the matrix that will hold the result
 * @param R_n_row   is the number of rows in the result
 * @param R_n_col   is the number of columns in the result
 */
void matrix_mul(double *A, int A_n_row, int A_n_col, double*B, int B_n_row, int B_n_col, double*R, int R_n_row, int R_n_col) {
    int Rij;

    for (int i = 0; i < A_n_row; i++) {
        for (int j = 0; j < B_n_col; j++) {
            Rij = i * R_n_col + j;
            R[Rij] = 0;
            for (int k = 0; k < A_n_col; k++) {
                R[Rij] += A[i * A_n_col + k] * B[k * B_n_col + j];
            }
        }
    }
}

/**
 * @brief compute the multiplication of A^T and B
 * @param A         is the matrix to be transposed
 * @param A_n_row   is the number of rows in matrix A
 * @param A_n_col   is the number of columns in matrix A
 * @param B         is the other factor of the multiplication
 * @param B_n_row   is the number of rows in matrix B
 * @param B_n_col   is the number of columns in matrix B
 * @param R         is the matrix that will hold the result
 * @param R_n_row   is the number of rows in the result
 * @param R_n_col   is the number of columns in the result
 */
void matrix_ltrans_mul(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col) {

    int Rij;

    for (int i = 0; i < A_n_col; i++) {
        for (int j = 0; j < B_n_col; j++) {
            Rij = i * R_n_col + j;
            R[Rij] = 0;
            for (int k = 0; k < B_n_row; k++)
                R[Rij] += A[k * A_n_col + i] * B[k * B_n_col + j];
        }
    }
}

/**
 * @brief compute the multiplication of A and B^T
 * @param A         is the other factor of the multiplication
 * @param A_n_row   is the number of rows in matrix A
 * @param A_n_col   is the number of columns in matrix A
 * @param B         is the matrix to be transposed
 * @param B_n_row   is the number of rows in matrix B
 * @param B_n_col   is the number of columns in matrix B
 * @param R         is the matrix that will hold the result
 * @param R_n_row   is the number of rows in the result
 * @param R_n_col   is the number of columns in the result
 */
void matrix_rtrans_mul(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col) {
    
    int Rij;

    for (int i = 0; i < A_n_row; i++) {
        for (int j = 0; j < B_n_row; j++) {
            Rij = i * R_n_col + j;
            R[Rij] = 0;
            for (int k = 0; k < A_n_col; k++)
                R[Rij] += A[i * A_n_col + k] * B[j * B_n_col + k];
        }
    }
}

/**
 * @brief computes the error based on the Frobenius norm 0.5*||V-WH||^2. The error is
 *        normalized with the norm V
 *
 * @param approx    is the matrix to store the W*H approximation
 * @param V         is the original matrix
 * @param W         is the first factorization matrix
 * @param H         is the second factorization matrix
 * @param m         is the number of rows in V
 * @param n         is the number of columns in V
 * @param r         is the factorization parameter
 * @param mn        is the number of elements in matrices V and approx
 * @param norm_V    is 1 / the norm of matrix V
 * @return          is the error
 */
inline double error(double* approx, double* V, double* W, double* H, int m, int n, int r, int mn, double norm_V) {

    matrix_mul(W, m, r, H, r, n, approx, m, n);

    double norm_approx, temp;

    norm_approx = 0;
    for (int i = 0; i < mn; i++)
    {
        temp = V[i] - approx[i];
        norm_approx += temp * temp;
    }
    norm_approx = sqrt(norm_approx);

    return norm_approx * norm_V;
}

/**
 * @brief computes the non-negative matrix factorisation updating the values stored by the 
 *        factorization functions
 * 
 * @param V             the matrix to be factorized
 * @param W             the first matrix in which V will be factorized
 * @param H             the second matrix in which V will be factorized
 * @param m             the number of rows of V
 * @param n             the number of columns of V
 * @param r             the factorization parameter
 * @param maxIteration  maximum number of iterations that can run
 * @param epsilon       difference between V and W*H that is considered acceptable
 */
double nnm_factorization(double *V, double*W, double*H, int m, int n, int r, int maxIteration, double epsilon) {

    int rn, rr, mr, mn;
    rn = r * n;
    rr = r * r;
    mr = m * r;
    mn = m * n;

    //Operands needed to compute Hn+1
    double *numerator, *denominator_l, *denominator;    //r x n, r x r, r x n
    numerator = malloc(double_size * rn);
    denominator_l = malloc(double_size * rr);
    denominator = malloc(double_size * rn);

    //Operands needed to compute Wn+1
    double *numerator_W, *denominator_W, *denominator_l_W;      // m x r, m x r, m x n
    numerator_W = malloc(double_size * mr);
    denominator_W = malloc(double_size * mr);
    denominator_l_W = malloc(double_size * mn);

    double* approximation; //m x n
    approximation = malloc(double_size * mn);

    double norm_V = 0;
    for (int i = 0; i < mn; i++)
        norm_V += V[i] * V[i];
    norm_V = 1 / sqrt(norm_V);

    //real convergence computation
    double err = -1;											
    for (int count = 0; count < maxIteration; count++) {
     
        err = error(approximation, V, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }

        //computation for Hn+1
        matrix_ltrans_mul(W, m, r, V, m, n, numerator, r, n);
        matrix_ltrans_mul(W, m, r, W, m, r, denominator_l, r, r);
        matrix_mul(denominator_l, r, r, H, r, n, denominator, r, n);

        for (int i = 0; i < rn; i++)
            H[i] = H[i] * numerator[i] / denominator[i];

        //computation for Wn+1
        matrix_rtrans_mul(V, m, n, H, r, n, numerator_W, m, r);
        matrix_mul(W, m, r, H, r, n, denominator_l_W, m, n);
        matrix_rtrans_mul(denominator_l_W, m, n, H, r, n, denominator_W, m, r);

        for (int i = 0; i < mr; i++)
            W[i] = W[i] * numerator_W[i] / denominator_W[i];
    }

    free(numerator);
    free(denominator);
    free(denominator_l);
    free(numerator_W);
    free(denominator_W);
    free(denominator_l_W);
    
    return err;
}

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

int main_opt(int argc, char const* argv[]) {

    double *V;      //m x n
    double *W, *H;  //m x r and r x n

    int m, n, r;
    m = 1000;
    n = 1000;
    r = 12;

    int mn, nr, mr;

    double rand_max_r = 1 / (double)RAND_MAX;

    if (argc != 2 && argc != 4) {
        printf("This program can be run in tow different modes:\n");
        printf("\t./ <m> <n> <r>\n");
        printf("\t./ <file-path>\n");
        return -1;
    }
    FILE* fp = NULL;
    if (argc == 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        r = atoi(argv[3]);

        srand(40);

        mn = m * n;
        V = malloc(double_size * mn);
        for (int i = 0; i < mn; i++)
            V[i] = rand() * rand_max_r;
    }

    if (argc == 2) {
        fp = fopen(argv[1], "r");
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

    double err = nnm_factorization(V, W, H, m, n, r, 10000, 0.005);
    printf("Error: %lf\n", err);

    free(V);
    free(W);
    free(H);

    return 0;
}