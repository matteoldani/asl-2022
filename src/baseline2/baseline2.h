#ifndef ASL_2022_BASELINE2_H
#define ASL_2022_BASELINE2_H

#include "../asl_utils/asl_utils.h"

double nnm_factorization_bs2(vMatrix *V, vMatrix *W, vMatrix *H, int maxIteration, double epsilon);
double nnm_factorization_bs2_cost(vMatrix * V, vMatrix * W, vMatrix * H, int numIterations);
long random_v_matrix_init_cost(vMatrix* matrix);
long random_v_acol_matrix_init_cost(vMatrix* V, vMatrix* W, int q);

#endif //ASL_2022_BASELINE2_H
