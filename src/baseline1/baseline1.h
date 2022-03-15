#ifndef ASL_2022_BASELINE1_H
#define ASL_2022_BASELINE1_H

#include "../asl_utils/asl_utils.h"

typedef struct {
    double **M;
    int n_row;
    int n_col;
} Matrix;

void random_matrix_init(Matrix *matrix, double min, double max);

void matrix_allocation(Matrix *matrix);

void matrix_deallocation(Matrix *matrix);

double nnm_factorization_bs1(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon);

#endif //ASL_2022_BASELINE1_H
