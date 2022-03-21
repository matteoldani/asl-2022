#ifndef ASL_2022_BASELINE1_H
#define ASL_2022_BASELINE1_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../asl_utils/asl_utils.h"

double nnm_factorization_bs1(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon);

#endif //ASL_2022_BASELINE1_H