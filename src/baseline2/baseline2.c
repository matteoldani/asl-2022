#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "cblas.h"
#include "baseline2.h"

void matrix_allocation(vMatrix *matrix);

void matrix_deallocation(vMatrix *matrix);

void random_matrix_init(vMatrix *matrix);

void matrix_mul(vMatrix *A, vMatrix *B, vMatrix *R);

void matrix_ltrans_mul(vMatrix *A, vMatrix *B, vMatrix *R);

void matrix_rtrans_mul(vMatrix *A, vMatrix *B, vMatrix *R);

void print_matrix(vMatrix *matrix);

double error(vMatrix *V, vMatrix *W, vMatrix *H);

double rand_from(double min, double max);

double norm(vMatrix *matrix);

/**
 * @brief allocates the matrix as an array inside the struct
 * @param matrix    is the struct where the matrix will be allocated
 */
void matrix_allocation(vMatrix *matrix) {

    // allocate the matrix dynamically
    matrix->M = malloc(sizeof(double *) * matrix->n_row * matrix->n_col);
}

/**
 * @brief deallocates the matrix
 * @param matrix    is the struct where the matrix will be deallocated
 */
void matrix_deallocation(vMatrix *matrix) {

    free(matrix->M);
}


// _____________________________ MATRIX MUL _____________________________
/**
 * @brief compute the multiplication of A and B
 * @param A is the first factor
 * @param B is the other factor of the multiplication
 * @param R is the matrix that will hold the result
 */

// RowMajor implementation
void matrix_mul_rm(vMatrix *A, vMatrix *B, vMatrix *R) {

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A->n_row, B->n_col, A->n_col, 1,
                A->M, A->n_col, B->M, B->n_col,
                0, R->M, B->n_col);
}

// Straightforward implementation (no BLAS)
void matrix_mul_s(vMatrix *A, vMatrix *B, vMatrix *R) {

    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i * R->n_col + j] += A->M[i * A->n_col + k] * B->M[k * B->n_col + j];

            }

        }
    }
}

// Working impl
void matrix_mul(vMatrix *A, vMatrix *B, vMatrix *R) {

    matrix_mul_rm(A, B, R);
}
//_____________________________________________________________________________


// _____________________________ LEFT MATRIX MUL _____________________________
/**
 * @brief compute the multiplication of A and B transposed
 * @param A is the other factor of the multiplication
 * @param B is the matrix to be transposed
 * @param R is the matrix that will hold the result
 */

// ColMajor impl  ----Param num 10 has an illegal value (0.270872)
// m=n=100        ----Param num 10 has an illegal value (nan)
void matrix_ltrans_mul_cm(vMatrix *A, vMatrix *B, vMatrix *R) {

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                A->n_col, B->n_col, A->n_row, 1,
                A->M, A->n_col, B->M, B->n_row,
                0, R->M, A->n_col);
}

// RowMajor impl (0.47)
void matrix_ltrans_mul_rm(vMatrix *A, vMatrix *B, vMatrix *R) {

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                A->n_col, B->n_col, B->n_row, 1, //r=A->n_row = B->n_row
                A->M, A->n_col, B->M, B->n_col,
                0, R->M, B->n_col);
}

// Straightforward implementation (no BLAS)
void matrix_ltrans_mul_s(vMatrix *A, vMatrix *B, vMatrix *R) {

    for (int i = 0; i < A->n_col; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < B->n_row; k++) {
                R->M[i * R->n_col + j] += A->M[k * A->n_col + i] * B->M[k * B->n_col + j];
            }
        }
    }
}

// Working impl
void matrix_ltrans_mul(vMatrix *A, vMatrix *B, vMatrix *R) {

    matrix_ltrans_mul_rm(A, B, R);
}
//_____________________________________________________________________________


// _____________________________ RIGHT MATRIX MUL _____________________________
/**
 * @brief compute the multiplication of A transposed and B
 * @param A is the matrix to be transposed
 * @param B is the other factor of the multiplication
 * @param R is the matrix that will hold the result
 */

// ColMajor impl  ----Param num 8 has an illegal value (0.327659)
void matrix_rtrans_mul_cm(vMatrix *A, vMatrix *B, vMatrix *R) {

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                A->n_row, B->n_row, A->n_col, 1,
                A->M, A->n_row, B->M, B->n_col,
                0, R->M, A->n_row);
}

// RowMajor impl  ----Param num 10 has an illegal value (0.475054)
void matrix_rtrans_mul_rm(vMatrix *A, vMatrix *B, vMatrix *R) {

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                A->n_row, B->n_row, A->n_col, 1,
                A->M, A->n_row, B->M, B->n_col,
                0, R->M, A->n_row);
}

//  Straightforward implementation (no BLAS)
void matrix_rtrans_mul_s(vMatrix *A, vMatrix *B, vMatrix *R) {

    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_row; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i * R->n_col + j] += A->M[i * A->n_col + k] * B->M[j * B->n_col + k];

            }
        }
    }
}

// Working impl
void matrix_rtrans_mul(vMatrix *A, vMatrix *B, vMatrix *R) {

    matrix_rtrans_mul_rm(A, B, R);
}
//_____________________________________________________________________________


/**
 * @brief prints the matrix
 * @param matrix    the matrix to be printed
 */
void print_matrix(vMatrix *matrix) {

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
void random_matrix_init(vMatrix *matrix) {

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
double nnm_factorization_bs2(vMatrix *V, vMatrix *W, vMatrix *H, int maxIteration, double epsilon) {
    int count = maxIteration;

    //Operands needed to compute Hn+1
    vMatrix numerator, denominator_l, denominator;

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
    vMatrix numerator_W, denominator_l_W, denominator_W;

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

        for (int i = 0; i < H->n_row * H->n_col; i++)
            H->M[i] = H->M[i] * numerator.M[i] / denominator.M[i];

        //computation for Wn+1
        matrix_rtrans_mul(V, H, &numerator_W);
        matrix_mul(W, H, &denominator_l_W);
        matrix_rtrans_mul(&denominator_l_W, H, &denominator_W);

        for (int i = 0; i < W->n_row * W->n_col; i++)
            W->M[i] = W->M[i] * numerator_W.M[i] / denominator_W.M[i];
    }

    matrix_deallocation(&numerator);
    matrix_deallocation(&denominator);
    matrix_deallocation(&denominator_l);
    matrix_deallocation(&numerator_W);
    matrix_deallocation(&denominator_W);
    matrix_deallocation(&denominator_l_W);
    return err;
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
double error(vMatrix *V, vMatrix *W, vMatrix *H) {

    vMatrix approximation;

    approximation.n_row = V->n_row;
    approximation.n_col = V->n_col;

    matrix_allocation(&approximation);

    matrix_mul(W, H, &approximation);

    double V_norm = norm(V);
    double approximation_norm;

    for (int i = 0; i < V->n_row * V->n_col; i++)
        approximation.M[i] = (V->M[i] - approximation.M[i]);

    approximation_norm = norm(&approximation);

    matrix_deallocation(&approximation);

    return approximation_norm / V_norm;
}


/**
 * @brief computes the frobenius norm of a matrix
 *
 * @param matrix is the matrix which norm is computed
 * @return the norm
 */
double norm(vMatrix *matrix) {
    double temp_norm = 0;
    for (int i = 0; i < matrix->n_row * matrix->n_col; i++)
        temp_norm += matrix->M[i] * matrix->M[i];

    return sqrt(temp_norm);
}

/**
 * @brief represents a dynamic allocated matrix
 * @param M     is the matrix
 * @param n_row is the number of rows
 * @param n_col is the number of cols
 */

int main(int argc, char const *argv[]) {

    vMatrix V;
    vMatrix W, H;
    int m = 100, n = 110, r = 12;

    srand(time(NULL));
    // read the desired factorization dimension
    //fscanf(stdin, "%d", &r);
    // read the dimensions
    //fscanf(stdin, "%d %d", &m, &n);

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

    random_matrix_init(&V);
    print_matrix(&V);

    nnm_factorization_bs2(&V, &W, &H, 100, 0.5);

    print_matrix(&W);
    print_matrix(&H);

    matrix_deallocation(&V);
    matrix_deallocation(&W);
    matrix_deallocation(&H);

    return 0;
}