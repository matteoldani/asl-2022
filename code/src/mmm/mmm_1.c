#include <mmm/mmm_1.h>

// matrix multi from baseline 2
void matrix_mul_1(Matrix *A, Matrix *B, Matrix *R) {

    // cost: B_col*A_col + 2*A_row*A_col*B_col
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A->n_row, B->n_col, A->n_col, 1,
                A->M, A->n_col, B->M, B->n_col,
                0, R->M, B->n_col);
}