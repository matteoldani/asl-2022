#include "../baseline1/baseline1.h"

#define NUM_RUNS 50
#define FREQUENCY 2.1e9		//change to fit your machine

int main(int argc, char const* argv[])
{
    Matrix V;
    Matrix W, H;
    int m, n, r;

    srand(time(NULL));

    long cost;
    double cycles, performance;
    clock_t start, end;
    int num_runs = NUM_RUNS;

    int numTests, min, max, steps;
    printf("Enter the number of tests to be executed: ");
    fscanf(stdin, "%d", &numTests);
    printf("Enter the minimal and maximal size of the matrices: ");
    fscanf(stdin, "%d %d", &min, &max);
    printf("\n");

    steps = (max - min) / numTests;
    for (int i = min; i < max; i += steps) {
        m = i;
        n = i + steps;
        r = 12;

        V.n_row = m;
        V.n_col = n;
        matrix_allocation(&V);

        W.n_row = m;
        W.n_col = r;
        matrix_allocation(&W);

        H.n_row = r;
        H.n_col = n;
        matrix_allocation(&H);

        //Call adequate cost functions
        cost = random_matrix_init_cost(&W);
        cost += random_matrix_init_cost(&H);
        cost += random_matrix_init_cost(&V);
        cost += nnm_factorization_bs1_cost(&V, &W, &H, 100);

        start = clock();
        for (int j = 0; j < num_runs; j++) {

            random_matrix_init(&W, 0, 1);
            random_matrix_init(&H, 0, 1);
            random_matrix_init(&V, 0, 1);

            nnm_factorization_bs1(&V, &W, &H, 100, 0.005);
        }
        end = clock();
        cycles = (double)(end - start) / num_runs;
        //cycles = (cycles / CLOCKS_PER_SEC) * FREQUENCY;
        //performance = cost / cycles;
        performance = ((cost / cycles) / FREQUENCY) * CLOCKS_PER_SEC;
        printf("Sizes: m=%d, n=%d, r=%d:\n", m, n, r);
        printf("--- cost(flops):%d, performance(flops/cycle):%f\n\n", cost, performance);

        matrix_deallocation(&V);
        matrix_deallocation(&W);
        matrix_deallocation(&H);
    }

    return 0;
}