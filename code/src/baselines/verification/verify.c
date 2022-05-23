#include <baselines/verification/verify.h>

void testRun(int m, int n, int r, int maxIteration, double epsilon, double min, double max) {
    double resultBS1, resultBS2;
    srand(SEED);

    Matrices matrices;
    Matrix W_temp;
    Matrix H_temp;

    generate_baseline_test_suite(&matrices, m, n, r, min, max);

    // copy the matrices 
    matrix_allocation(&W_temp, matrices.W.n_row, matrices.W.n_col);
    matrix_allocation(&H_temp, matrices.H.n_row, matrices.H.n_col);
    
    for(int i=0; i<matrices.W.n_col*matrices.W.n_row;i++){
        W_temp.M[i] = matrices.W.M[i];
    }

    for(int i=0; i<matrices.H.n_col*matrices.H.n_row;i++){
        H_temp.M[i] = matrices.H.M[i];
    }

    resultBS1 = nnm_factorization_bs1(&matrices.V, &matrices.W,
                                      &matrices.H, maxIteration, epsilon);

    resultBS2 = nnm_factorization_bs2(&matrices.V, &W_temp,
                                      &H_temp, maxIteration, epsilon);
    printf("Results: error_bs1=%lf, error_bs2=%lf\n", resultBS1, resultBS2);
    if (fabs(resultBS1 - resultBS2) > 0.000001) {
        printf("ERROR: Difference: %f\n", fabs(resultBS1 - resultBS2));
        printf("Parameters: m=%d, n=%d, r=%d\n", m, n, r);
    }
}

void generate_baseline_test_suite(Matrices *matrices, int m, int n, int r, double min, double max) {

    allocate_base_matrices(matrices, m, n, r);
    
    random_matrix_init(&matrices->V,min, max);
    random_matrix_init(&matrices->W,min, max);

    // random_acol_matrix_init(&b->bs1Matrices.V,&b->bs1Matrices.W, 3);
    // random_acol_matrix_init(&b->bs2Matrices.V,&b->bs2Matrices.W, 3);

    random_matrix_init(&matrices->H,min, max);

    // printf("****************** Matrix V ******************\n");
    // print_matrix(&matrices->V);
    // printf("****************** Matrix W ******************\n");
    // print_matrix(&matrices->W);
    // printf("****************** Matrix H ******************\n");
    // print_matrix(&matrices->H);

}


int main(int argc, char const *argv[]) {

    int numTests, min, max, steps, maxIteration, count;
    int m, n, r;
    double epsilon, low, high;
    
    printf("Those are the values required: \n");

    // read the number of tests to be executed
    printf("\tNum of Tests: ");
    fscanf(stdin, "%d", &numTests);
    
    // read the minimal and maximal size of the matrices
    printf("\tMin & Max: ");
    fscanf(stdin, "%d %d", &min, &max);

    // read the maxIteration
    printf("\tMax Iterations: ");
    fscanf(stdin, "%d", &maxIteration);
    // read the epsilon

    printf("\tEpsilon: ");
    fscanf(stdin, "%lf", &epsilon);

    // read the low and high of the values in the matrices
    printf("\tLow & High: ");
    fscanf(stdin, "%lf %lf", &low, &high);


    steps = (max - min) / numTests;
    count = 0;
    for (int i = min; i < max; i += steps) {
        //printf("Starting iteration: %d\n", i/steps);
        m = i;
        n = i + steps;
        r = 12;
        testRun(m, n, r, maxIteration, epsilon, low, high);
        count++;
    }
}