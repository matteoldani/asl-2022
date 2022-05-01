#ifndef ASL_2022_BASELINE2_H
#define ASL_2022_BASELINE2_H

#include <baselines/baselines_utils.h>


void matrix_mul_bs2(Matrix *A, Matrix *B, Matrix *R);
void matrix_ltrans_mul_bs2(Matrix *A, Matrix *B, Matrix *R);
void matrix_rtrans_mul_bs2(Matrix *A, Matrix *B, Matrix *R);
double error_bs2(Matrix *V, Matrix *W, Matrix *H);
double nnm_factorization_bs2(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon);

myInt64 nnm_factorization_bs2_cost(Matrix * V, Matrix * W, Matrix * H, int numIterations);
myInt64 random_v_matrix_init_cost(Matrix* matrix);
myInt64 random_v_acol_matrix_init_cost(Matrix* V, Matrix* W, int q);

#endif //ASL_2022_BASELINE2_H
