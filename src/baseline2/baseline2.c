#include <stdlib.h>
#include <stdio.h>
#include "lapacke.h"
#include "lapacke_config.h"
#include <time.h>
#include <math.h>
#include "cblas.h"

/* Parameters */
#define N 5
#define NRHS 3
#define LDA N
#define LDB NRHS
#define EPSILON 0.5
#define MAX_ITERATION  500

/**
 * @brief represents a dynamic allocated matrix
 * @param M     is the matrix
 * @param n_row is the number of rows
 * @param n_col is the number of cols
 */
typedef struct {
    double *M;
    int n_row;
    int n_col;
} Matrix;

void matrix_allocation(Matrix *matrix);

void read_input(Matrix *matrix);

void random_matrix_init(Matrix *matrix);

void nnm_factorization(Matrix *V, Matrix *W, Matrix *H);

void print_matrix(Matrix *matrix);

double error(Matrix *V, Matrix *W, Matrix *H);

double rand_from(double min, double max);

double norm(Matrix *matrix);

int main(int argc, char const *argv[]) {

    Matrix V;
    Matrix W, H;
    int m, n, r;

    srand(time(NULL));
    // read the desired factorization dimension
    fscanf(stdin, "%d", &r);
    // read the dimensions
    fscanf(stdin, "%d %d", &m, &n);

    V.n_row = m;
    V.n_col = n;
    matrix_allocation(&V);

    W.n_row = m;
    W.n_col = r;
    matrix_allocation(&W);

    H.n_row = r;
    H.n_col = n;
    matrix_allocation(&H);

    random_matrix_init(&W);
    random_matrix_init(&H);

    read_input(&V);
    print_matrix(&V);

    nnm_factorization(&V, &W, &H);

    print_matrix(&W);
    print_matrix(&H);

    return 0;
}

/**
 * @brief allocates the matrix as an array inside the struct
 * @param matrix    is the struct where the matrix will be allocated
 */
void matrix_allocation(Matrix *matrix) {

    // allocate the matrix dynamically
    matrix->M = malloc(sizeof(double *) * matrix->n_row * matrix->n_col);
}

/**
 * @brief reads the input corresponfind to the matrix values
 * @param matrix    the matrix that will be filled
 */
void read_input(Matrix *matrix) {

    for (int i = 0; i < matrix->n_row * matrix->n_col; i++)
        fscanf(stdin, "%lf", &(matrix->M[i]));
}

/**
 * @brief prints the matrix
 * @param matrix    the matrix to be printed
 */
void print_matrix(Matrix *matrix) {

    printf("Printing a matrix with %d rows and %d cols\n\n", matrix->n_row, matrix->n_col);
    for (int row = 0; row < matrix->n_row; row++) {
        for (int col = 0; col < matrix->n_col; col++) {
            fprintf(stdout, "%.2lf\t", matrix->M[row * matrix->n_col + col]);
        }
        fprintf(stdout, "\n\n");
    }

    fprintf(stdout, "\n\n");
}

/**
 * @brief generate a random floating point number from min to max
 * @param min   the minumum possible value
 * @param max   the maximum possible value
 * @return      the random value
 */
double rand_from(double min, double max) {

    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}


/**
 * @brief initialize a matrix with random numbers between 0 and 1
 * @param matrix    the matrix to be initialized
 */
void random_matrix_init(Matrix *matrix) {

    for (int i = 0; i < matrix->n_row * matrix->n_col; i++)
        matrix->M[i] = rand_from(0.00, 1.00);
}

/**
 * @brief computes the non-negative matrix factorisation updating the values stored by the
 *        factorization functions
 *
 * @param V     the matrix to be factorized
 * @param W     the first matrix in which V will be factorized
 * @param H     the second matrix in which V will be factorized
 */
void nnm_factorization(Matrix *V, Matrix *W, Matrix *H) {

    //Operands needed to compute Hn+1
    Matrix numerator, denominator_l, denominator;

    numerator.n_row = W->n_col;
    numerator.n_col = V->n_col;

    denominator_l.n_row = W->n_col;
    denominator_l.n_col = W->n_col;

    denominator.n_row = H->n_row;
    denominator.n_col = H->n_col;

    matrix_allocation(&numerator);
    matrix_allocation(&denominator);
    matrix_allocation(&denominator_l);

    //Operands needed to compute Wn+1
    Matrix numerator_W, denominator_l_W, denominator_W;

    numerator_W.n_row = V->n_row;
    numerator_W.n_col = H->n_row;

    denominator_l_W.n_row = W->n_row;
    denominator_l_W.n_col = H->n_col;

    denominator_W.n_row = W->n_row;
    denominator_W.n_col = W->n_col;

    matrix_allocation(&numerator_W);
    matrix_allocation(&denominator_W);
    matrix_allocation(&denominator_l_W);

    //real convergence computation
    double err;
    int count = MAX_ITERATION;
    while (--count /*(err = error(V, W, H)) > EPSILON */) {
        err = error(V, W, H);
        printf("Current error: %lf\n", err);

        //computation for Wn+1
        cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasTrans, W->n_row, V->n_col, W->n_col, 1, W->M, W->n_row, V->M, V->n_col, 0, numerator.M,
                      numerator.n_row);
        // matrix_ltrans_mul(W, V, &numerator);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, W->n_row, W->n_col, W->n_col, 1, W->M, W->n_row, W->M, W->n_col, 0, denominator_l.M,
                      denominator_l.n_row);
        //matrix_ltrans_mul(W, W, &denominator_l);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, denominator_l.n_row, H->n_col, denominator_l.n_col, 1, denominator_l.M,
                      denominator_l.n_row, H->M, H->n_row, 0, denominator.M, denominator.n_row);
        //matrix_mul(&denominator_l, H, &denominator);

        for (int i = 0; i < H->n_row * H->n_col; i++)
            H->M[i] = H->M[i] * numerator.M[i] / denominator.M[i];

        //computation for Wn+1
        cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, V->n_row, H->n_col, V->n_col, 1, V->M, V->n_col, H->M, H->n_row, 0, numerator_W.M,
                      numerator_W.n_row);
        //matrix_rtrans_mul(V, H, &numerator_W);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, W->n_row, H->n_col, W->n_col, 1, W->M, W->n_row, H->M, H->n_row, 0, denominator_l_W.M,
                      denominator_l_W.n_row);
        //matrix_mul(W, H, &denominator_l_W);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, denominator_l_W.n_row, H->n_col, denominator_l_W.n_col, 1, denominator_l_W.M,
                      denominator_l_W.n_col, H->M, H->n_row, 0, denominator_W.M, denominator_W.n_row);
        //matrix_rtrans_mul(&denominator_l_W, H, &denominator_W);

        for (int i = 0; i < W->n_row * W->n_col; i++)
            W->M[i] = W->M[i] * numerator_W.M[i] / denominator_W.M[i];
    }
}

/**
 * @brief computes the error based on the Frobenius norm 0.5*||V-WH||^2. The error is
 *        normalized with the norm V
 *
 * @param V is the original matrix
 * @param W is the first factorization matrix
 * @param H is the second factorization matrix
 * @return is the error
 */
double error(Matrix *V, Matrix *W, Matrix *H) {

    Matrix approximation;

    approximation.n_row = V->n_row;
    approximation.n_col = V->n_col;

    matrix_allocation(&approximation);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, W->n_row, H->n_col, W->n_col, 1, W->M, W->n_row, H->M, H->n_row, 0, approximation.M,
                  approximation.n_row);
    //matrix_mul(W, H, &approximation);

    double V_norm = norm(V);
    double approximation_norm;

    for (int i = 0; i < V->n_row * V->n_col; i++)
        approximation.M[i] = (V->M[i] - approximation.M[i]);

    approximation_norm = norm(&approximation);
    return approximation_norm / V_norm;
}


/**
 * @brief computes the frobenius norm of a matrix
 *
 * @param matrix is the matrix which norm is computed
 * @return the norm
 */
double norm(Matrix *matrix) {
    double temp_norm = 0;
    for (int i = 0; i < matrix->n_row * matrix->n_col; i++)
        temp_norm += matrix->M[i] * matrix->M[i];

    return sqrt(temp_norm);
}