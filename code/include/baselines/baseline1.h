#ifndef ASL_2022_BASELINE1_H
#define ASL_2022_BASELINE1_H

#include <baselines/baselines_utils.h>


void matrix_mul_bs1(Matrix *A, Matrix *B, Matrix *R);
void matrix_ltrans_mul_bs1(Matrix *A, Matrix *B, Matrix *R);
void matrix_rtrans_mul_bs1(Matrix *A, Matrix *B, Matrix *R);
double error_bs1(Matrix *V, Matrix *W, Matrix *H);
double nnm_factorization_bs1(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon);

myInt64 random_matrix_init_cost(Matrix* matrix);
myInt64 random_acol_matrix_init_cost(Matrix* V, Matrix* W, int q);
myInt64 nnm_factorization_bs1_cost(Matrix* V, Matrix* W, Matrix* H, int numIterations);

#endif //ASL_2022_BASELINE1_H
