#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "baseline1.h"

void matrix_allocation(Matrix *matrix);

void read_input(Matrix *matrix);

void random_matrix_init(Matrix *matrix);

void random_acol_matrix_init(Matrix *V, Matrix *W, int q);

void matrix_mul(Matrix *A, Matrix *B, Matrix *R);

void matrix_ltrans_mul(Matrix *A, Matrix *B, Matrix *R);

void matrix_rtrans_mul(Matrix *A, Matrix *B, Matrix *R);

void print_matrix(Matrix *matrix);

double error(Matrix *V, Matrix *W, Matrix *H);

double rand_from(double min, double max);

double norm(Matrix *matrix);

/**
 * @brief allocates the matrix as a double pointer inside the struct
 * @param matrix    is the struct where the matrix will be allocated
 */
void matrix_allocation(Matrix *matrix) {

    // allocate the matrix dynamically 
    matrix->M = malloc(sizeof(double *) * matrix->n_row);

    for (int row = 0; row < matrix->n_row; row++)
        (matrix->M)[row] = malloc(sizeof(double) * matrix->n_col);
}

/**
 * @brief reads the input corresponfind to the matrix values
 * @param matrix    the matrix that will be filled
 */
void read_input(Matrix *matrix) {

    for (int row = 0; row < matrix->n_row; row++) {
        for (int col = 0; col < matrix->n_col; col++)
            fscanf(stdin, "%lf", &(matrix->M[row][col]));
    }
}

/**
 * @brief compute the multiplication of A and B
 * @param A is the first factor 
 * @param B is the other factor of the multiplication
 * @param R is the matrix that will hold the result
 */
void matrix_mul(Matrix *A, Matrix *B, Matrix *R) {

    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i][j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i][j] += A->M[i][k] * B->M[k][j];
            }
        }
    }
}

/**
 * @brief compute the multiplication of A^T and B
 * @param A is the matrix to be transposed
 * @param B is the other factor of the multiplication
 * @param R is the matrix that will hold the result
 */
void matrix_ltrans_mul(Matrix *A, Matrix *B, Matrix *R) {

    for (int i = 0; i < A->n_col; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i][j] = 0;
            for (int k = 0; k < B->n_row; k++) {
                R->M[i][j] += A->M[k][i] * B->M[k][j];

            }
        }
    }
}

/**
 * @brief compute the multiplication of A and B^T
 * @param A is the other factor of the multiplication
 * @param B is the matrix to be transposed
 * @param R is the matrix that will hold the result
 */
void matrix_rtrans_mul(Matrix *A, Matrix *B, Matrix *R) {

    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_row; j++) {
            R->M[i][j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i][j] += A->M[i][k] * B->M[j][k];
            }
        }
    }
}

/**
 * @brief prints the matrix
 * @param matrix    the matrix to be printed
 */
void print_matrix(Matrix *matrix) {

    printf("Printing a matrix with %d rows and %d cols\n\n", matrix->n_row, matrix->n_col);
    for (int row = 0; row < matrix->n_row; row++) {
        for (int col = 0; col < matrix->n_col; col++) {
            fprintf(stdout, "%.2lf\t", matrix->M[row][col]);
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

    for (int row = 0; row < matrix->n_row; row++) {
        for (int col = 0; col < matrix->n_col; col++) {
            matrix->M[row][col] = rand_from(0.00, 1.00);
        }
    }
}


/**
 * @brief initialize a matrix W averaging columns of X
 * @param V    matrix to be factorized
 * @param W    factorizing matrix, initialized here
 * @param q    number of columns of X averaged to obtsain a column of W 
 */
void random_acol_matrix_init(Matrix *V, Matrix *W, int q) {
    int r;

    // initialize W to all zeros
    for(int k = 0; k < W -> n_row; k++)
        memset(W->M[k], 0,  sizeof(double) * W->n_col);

    for(int  k = 0; k < W -> n_col; k++){
        //average q random column of X into W

        for (int i = 0; i < q; i++){
            r = rand() % V->n_col; 
            for (int j = 0; j < V -> n_row; j++)
                W->M[j][k] += V->M[j][r];     
        }

        for (int j = 0; j < V -> n_row; j++)
            W->M[j][k] = W->M[j][k] / q;   
    }
}





/**
 * @brief computes the non-negative matrix factorisation updating the values stored by the 
 *        factorization functions
 * 
 * @param V     the matrix to be factorized
 * @param W     the first matrix in which V will be factorized
 * @param H     the second matrix in which V will be factorized
 */
double nnm_factorization_bs1(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon) {
    int count = maxIteration;

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
    err = error(V, W, H);
    for (;;) {
        if (maxIteration > 0 && count == 0) {
            break;
        }
        if (err <= epsilon) {
            break;
        }
        count--;
        err = error(V, W, H);
        printf("Current error: %lf\n", err);

        //computation for Hn+1
        matrix_ltrans_mul(W, V, &numerator);
        matrix_ltrans_mul(W, W, &denominator_l);
        matrix_mul(&denominator_l, H, &denominator);

        for (int i = 0; i < H->n_row; i++) {
            for (int j = 0; j < H->n_col; j++) {
                H->M[i][j] = H->M[i][j] * numerator.M[i][j] / denominator.M[i][j];
            }
        }

        //computation for Wn+1
        matrix_rtrans_mul(V, H, &numerator_W);
        matrix_mul(W, H, &denominator_l_W);
        matrix_rtrans_mul(&denominator_l_W, H, &denominator_W);

        for (int i = 0; i < W->n_row; i++) {
            for (int j = 0; j < W->n_col; j++) {
                W->M[i][j] = W->M[i][j] * numerator_W.M[i][j] / denominator_W.M[i][j];
            }
        }
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
    matrix_mul(W, H, &approximation);

    double V_norm = norm(V);
    double approximation_norm;

    for (int row = 0; row < V->n_row; row++)
        for (int col = 0; col < V->n_col; col++) {
            approximation.M[row][col] = (V->M[row][col] - approximation.M[row][col]);
        }

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

    for (int row = 0; row < matrix->n_row; row++) {
        for (int col = 0; col < matrix->n_col; col++) {
            temp_norm += matrix->M[row][col] * matrix->M[row][col];
        }
    }

    return sqrt(temp_norm);
}

/**
 * @brief represents a dynamic allocated matrix
 * @param M     is the matrix
 * @param n_row is the number of rows
 * @param n_col is the number of cols
 */

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

    read_input(&V);
    print_matrix(&V);

    //random_matrix_init(&W);
    random_acol_matrix_init(&V, &W, 3);
    random_matrix_init(&H);

    nnm_factorization_bs1(&V, &W, &H, 100, 0.5);

    print_matrix(&W);
    print_matrix(&H);
    printf("Error: %lf\n", error(&V,&W,&H));
    return 0;
}