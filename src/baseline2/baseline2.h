#ifndef ASL_2022_BASELINE2_H
#define ASL_2022_BASELINE2_H

#include "../asl_utils/asl_utils.h"

typedef struct {
    double *M;
    int n_row;
    int n_col;
} vMatrix;

void random_v_matrix_init(vMatrix *matrix, double min, double max);

void v_matrix_allocation(vMatrix *matrix);

void v_matrix_deallocation(vMatrix *matrix);

double nnm_factorization_bs2(vMatrix *V, vMatrix *W, vMatrix *H, int maxIteration, double epsilon);

#endif //ASL_2022_BASELINE2_H
