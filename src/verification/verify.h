#ifndef ASL_2022_VERIFY_H
#define ASL_2022_VERIFY_H

#include "../baseline2/baseline2.h"
#include "../baseline1/baseline1.h"

typedef struct {
    Baseline1Matrices bs1Matrices;
    Baseline2Matrices bs2Matrices;
} BaselineTestsuite;

void generate_baseline_test_suite(BaselineTestsuite *b, int m, int n, int r, double min, double max);

#endif //ASL_2022_VERIFY_H
