#include "verify.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "math.h"

void testRun(int m, int n, int r, int maxIteration, double epsilon, double min, double max) {
    double resultBS1, resultBS2;
    srand(time(NULL));

    BaselineTestsuite baselineTestsuite;
    generate_baseline_test_suite(&baselineTestsuite, m, n, r, min, max);

    resultBS1 = nnm_factorization_bs1(&baselineTestsuite.bs1Matrices.V, &baselineTestsuite.bs1Matrices.W,
                                      &baselineTestsuite.bs1Matrices.H, maxIteration, epsilon);
    resultBS2 = nnm_factorization_bs2(&baselineTestsuite.bs2Matrices.V, &baselineTestsuite.bs2Matrices.W,
                                      &baselineTestsuite.bs2Matrices.H, maxIteration, epsilon);
    printf("Results: error_bs1=%lf, error_bs2=%lf\n", resultBS1, resultBS2);
    printf("Difference: %f\n", fabs(resultBS1 - resultBS2));


}

void generate_baseline_test_suite(BaselineTestsuite *b, int m, int n, int r, double min, double max) {

    initialise_bs1_matrices(&b->bs1Matrices, m, n, r);
    initialise_bs2_matrices(&b->bs2Matrices, m, n, r);

    generate_random_matrices(&b->bs1Matrices.V, &b->bs2Matrices.V, min, max);
    generate_random_matrices(&b->bs1Matrices.W, &b->bs2Matrices.W, min, max);
    generate_random_matrices(&b->bs1Matrices.H, &b->bs2Matrices.H, min, max);
}

void deallocate_matrices(BaselineTestsuite *b) {

    matrix_deallocation(&b->bs1Matrices.V);
    matrix_deallocation(&b->bs1Matrices.W);
    matrix_deallocation(&b->bs1Matrices.H);
    v_matrix_deallocation(&b->bs2Matrices.V);
    v_matrix_deallocation(&b->bs2Matrices.W);
    v_matrix_deallocation(&b->bs2Matrices.H);
}

int main(int argc, char const *argv[]) {

    int numTests, min, max, steps, maxIteration, count;
    int m, n, r;
    double epsilon, low, high;
    // read the number of tests to be executed
    fscanf(stdin, "%d", &numTests);
    // read the minimal and maximal size of the matrices
    fscanf(stdin, "%d %d", &min, &max);
    // read the maxIteration
    fscanf(stdin, "%d", &maxIteration);
    // read the epsilon
    fscanf(stdin, "%lf", &epsilon);
    // read the low and high of the values in the matrices
    fscanf(stdin, "%lf %lf", &low, &high);

    steps = (max - min) / numTests;
    count = 0;
    for (int i = min; i < max; i += steps) {
        m = i;
        n = i + steps;
        r = 12;
        printf("Test run nr: %d\n", count);
        printf("Parameters: m=%d, n=%d, r=%d\n", m, n, r);
        testRun(m, n, r, maxIteration, epsilon, low, high);
        count++;
    }
}