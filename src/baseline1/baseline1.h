#ifndef ASL_2022_BASELINE1_H
#define ASL_2022_BASELINE1_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../asl_utils/asl_utils.h"

double nnm_factorization_bs1(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon);

long random_matrix_init_cost(Matrix* matrix);
long random_acol_matrix_init_cost(Matrix* V, Matrix* W, int q);
double nnm_factorization_bs1_cost(Matrix* V, Matrix* W, Matrix* H, int numIterations);

#endif //ASL_2022_BASELINE1_H
