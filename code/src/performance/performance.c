
//#error Please comment out the next two lines under linux, then comment this error
//#include "stdafx.h"  //Visual studio expects this line to be the first one, comment out if different compiler
//#include <windows.h> // Include if under windows

#include <performance/performance.h>

void baseline(int numTests, int min, int max, int b, FILE * fout, fact_function fact_function, fact_cost fact_cost_function){
    Matrix V;
    Matrix W, H;
    myInt64 m, n, r;
    printf("Baseline %d performance evaluation\n\n", b);
    srand(SEED);

    myInt64 cost;
    double performance;
    
    int num_runs = NUM_RUNS;

    int steps = (max - min) / numTests;
    for (int i = min; i < max; i += steps) {
        m = i;
        n = i; //+ steps; 
        r = 12;

        matrix_allocation(&V, m, n);
        matrix_allocation(&W, m, r);
        matrix_allocation(&H, r, n);

        //Call adequate cost functions

        cost = random_matrix_init_cost(&W);
        cost += random_matrix_init_cost(&H);
        cost += random_matrix_init_cost(&V);

        cost += fact_cost_function(&V, &W, &H, 100);

        #ifdef __x86_64__
        myInt64 cycles;
        myInt64 start;
        #ifdef CALIBRATE
        while(num_runs < (1 << 14)) {
            start = start_tsc();
            for (int j = 0; j < num_runs; j++) {

                random_matrix_init(&W, 0, 1);
                random_matrix_init(&H, 0, 1);
                random_matrix_init(&V, 0, 1);

                fact_function(&V, &W, &H, 100, 0.005);
            }
            cycles = stop_tsc(start);

            if(cycles >= CYCLES_REQUIRED) break;

            num_runs *= 2;
        }
        #endif

        start = start_tsc();
        for (int j = 0; j < num_runs; j++) {

            random_matrix_init(&W, 0, 1);
            random_matrix_init(&H, 0, 1);
            random_matrix_init(&V, 0, 1);

            fact_function(&V, &W, &H, 100, 0.005);
        }

        cycles = stop_tsc(start)/num_runs;
        performance =  (double )cost / (double) cycles;

        #endif

        #ifndef __x86_64__
        double cycles;
        clock_t start, end;
        #ifdef CALIBRATE
        while(num_runs < (1 << 14)) {
            start = clock();
            for (int j = 0; j < num_runs; j++) {

                random_matrix_init(&H, 0, 1);
                random_matrix_init(&W, 0, 1);
                random_matrix_init(&V, 0, 1);

                nnm_factorization_bs1(&V, &W, &H, 100, 0.005);
            }
            end = clock();

            cycles = (double)(end-start);

            // Same as in c_clock: CYCLES_REQUIRED should be expressed accordingly to the order of magnitude of CLOCKS_PER_SEC
            if(cycles >= CYCLES_REQUIRED/(FREQUENCY/CLOCKS_PER_SEC)) break;

            num_runs *= 2;
        }
        #endif

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
        
        #endif

        matrix_deallocation(&V);
        matrix_deallocation(&W);
        matrix_deallocation(&H);


        printf("Sizes: m=%llu, n=%llu, r=%llu:\n", m, n, r);
        printf("--- cost(flops):%llu, cycles:%llu, performance(flops/cycle):%lf\n\n", cost, cycles, performance);
         if(fout != NULL){
            fprintf(fout, "%llu,%llu,%llu,%llu,%lf\n",m,r,n,cost, performance);
        }
    }
}

int main(int argc, char const* argv[])
{
    if(argc <= 1){
        printf("How to use this tool:\n");
        printf("/<path>/<to>/<binary> ");
        printf("<baseline number [1,2]> ");
        printf("<min size matrix> ");
        printf("<max size matrix> ");
        printf("<number of test>");
        printf("<output_file>[?]\n\n");
        return 0;
    }

    int b, tests, min, max;
    FILE *fout = NULL;
    b = atoi(argv[1]);
    
    min = atoi(argv[2]);
    max = atoi(argv[3]);
    tests = atoi(argv[4]);

    if (argc > 5){
        fout = fopen(argv[5], "w+");
        if (fout == NULL){
            printf("Can't open output file\n");
            exit(-1);
        }
        fprintf(fout, "m,r,n,cycles,performance\n");
    }

    switch(b){
    case 1:
        baseline(tests, min, max, b, fout, &nnm_factorization_bs1, &nnm_factorization_bs1_cost);
        break;
    
    case 2:
        baseline(tests, min, max, b, fout, &nnm_factorization_bs2, &nnm_factorization_bs2_cost);
        break;
    
    default:
        break;
    }
    if( fout != NULL)
        fclose(fout);

    return 0;
}



