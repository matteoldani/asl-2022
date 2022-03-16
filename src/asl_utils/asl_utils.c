#include <stdlib.h>
#include "asl_utils.h"

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
 * @brief translate a 2D matrix into a vector matrix
 * @param matrix    the 2D matrix to be translated
 * @param vmatrix   the vector matrix to be translated into
 */
void translate_matrix_to_v_matrix(Matrix *matrix, vMatrix *vmatrix) {
    for (int row = 0; row < matrix->n_row; row++) {
        for (int col = 0; col < matrix->n_col; col++) {
            vmatrix->M[row * vmatrix->n_col + col] = matrix->M[row][col];
        }
    }
}

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
 * @brief deallocates the matrix
 * @param matrix    is the struct where the matrix will be deallocated
 */
void matrix_deallocation(Matrix *matrix) {

    free(matrix->M);
}

/**
 * @brief allocates the matrix as an array inside the struct
 * @param matrix    is the struct where the matrix will be allocated
 */
void v_matrix_allocation(vMatrix *matrix) {

    // allocate the matrix dynamically
    matrix->M = malloc(sizeof(double *) * matrix->n_row * matrix->n_col);
}

/**
 * @brief deallocates the matrix
 * @param matrix    is the struct where the matrix will be deallocated
 */
void v_matrix_deallocation(vMatrix *matrix) {

    free(matrix->M);
}

void initialise_bs1_matrices(Baseline1Matrices *bs1, int m, int n, int r) {
    bs1->V.n_row=m;
    bs1->V.n_col=n;
    matrix_allocation(&bs1->V);

    bs1->W.n_row=m;
    bs1->W.n_col=r;
    matrix_allocation(&bs1->W);

    bs1->H.n_row=r;
    bs1->H.n_col=n;
    matrix_allocation(&bs1->H);
}

void initialise_bs2_matrices(Baseline2Matrices *bs2, int m, int n, int r) {
    bs2->V.n_row=m;
    bs2->V.n_col=n;
    v_matrix_allocation(&bs2->V);

    bs2->W.n_row=m;
    bs2->W.n_col=r;
    v_matrix_allocation(&bs2->W);

    bs2->H.n_row=r;
    bs2->H.n_col=n;
    v_matrix_allocation(&bs2->H);
}

void allocate_matrix_v_matrix(Matrix *matrix, vMatrix *vmatrix) {
    matrix_allocation(matrix);
    v_matrix_allocation(vmatrix);
}

void generate_random_matrices(Matrix *matrix, vMatrix *vmatrix, double min, double max) {
    random_matrix_init(matrix, min, max);
    translate_matrix_to_v_matrix(matrix, vmatrix);
}
