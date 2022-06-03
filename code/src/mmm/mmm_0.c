#include <mmm/mmm_0.h>

// martix mult from baseline 1
void matrix_mul_0(Matrix *A, Matrix *B, Matrix *R) {            // 2 * A->n_row * B->n_col * A->n_col

    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i * R->n_col + j] += A->M[i * A->n_col + k] * B->M[k * B->n_col + j];
            }
        }
    }
}
