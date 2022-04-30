#ifndef ASL_2022_VERIFY_H
#define ASL_2022_VERIFY_H

#include <baselines/baseline1.h>
#include <baselines/baseline2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct {
    Matrices bs1Matrices;
    Matrices bs2Matrices;
} BaselineTestsuite;

void generate_baseline_test_suite(BaselineTestsuite *b, int m, int n, int r, double min, double max);

#endif //ASL_2022_VERIFY_H
