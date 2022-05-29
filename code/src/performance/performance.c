
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
    int max_iterations = MAX_ITERATIONS;

    int steps = (max - min) / numTests;
    for (int i = min; i < max; i += steps) {
        m = i;
        n = i; //+ steps; 
        r = RANK;

        matrix_allocation(&V, m, n);
        matrix_allocation(&W, m, r);
        matrix_allocation(&H, r, n);

        //Call adequate cost functions

        cost = random_matrix_init_cost(&W);
        cost += random_matrix_init_cost(&H);
        cost += random_matrix_init_cost(&V);

        cost += fact_cost_function(&V, &W, &H, max_iterations);

        #ifdef __x86_64__
        myInt64 cycles;
        myInt64 start;
        #ifdef CALIBRATE
        while(num_runs < (1 << 7)) {
            start = start_tsc();
            for (int j = 0; j < num_runs; j++) {

                random_matrix_init(&W, 0, 1);
                random_matrix_init(&H, 0, 1);
                random_matrix_init(&V, 0, 1);

                fact_function(&V, &W, &H, max_iterations, 0.005);
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

            fact_function(&V, &W, &H, max_iterations, 0.005);
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

void optimization(int numTests, int min, int max, int opt, FILE * fout, opt_fact_function fact_function, opt_fact_cost fact_cost_function){
    double* V;
    double* W;
    double* H;
    myInt64 m, n, r;
    printf("Opt alg %d performance evaluation\n\n", opt);
    srand(SEED);

    myInt64 cost;
    double performance;
    
    int num_runs = NUM_RUNS;
    int max_iterations = MAX_ITERATIONS;
    double rand_max_r = 1 / (double)RAND_MAX;

    int steps = (max - min) / numTests;
    for (int i = min; i < max; i += steps) {
        m = i;
        n = i; //+ steps; 
        r = RANK;
        V = malloc(m * n * sizeof(double));
        W = malloc(m * r * sizeof(double));
        H = malloc(r * n * sizeof(double));
      
        //Call adequate cost functions

        cost += matrix_rand_init_cost(n, m);
        cost = matrix_rand_init_cost(m, r);
        cost += matrix_rand_init_cost(r, n);

        cost += fact_cost_function(m, n, m, r, r, n, max_iterations);

        #ifdef __x86_64__
        myInt64 cycles;
        myInt64 start;
        #ifdef CALIBRATE
        while(num_runs < (1 << 14)) {
            start = start_tsc();
            for (int j = 0; j < num_runs; j++) {
                for (int i = 0; i < m*r; i++) W[i] = rand() * rand_max_r;
                for (int i = 0; i < n*r; i++) H[i] = rand() * rand_max_r;
                for (int i = 0; i < m*n; i++) V[i] = rand() * rand_max_r;

                fact_function(V, W, H, m, n, r, max_iterations, 0.005);
            }
            cycles = stop_tsc(start);

            if(cycles >= CYCLES_REQUIRED) break;

            num_runs *= 2;
        }
        #endif 

        start = start_tsc();
        for (int j = 0; j < num_runs; j++) {

            for (int i = 0; i < m*r; i++) W[i] = rand() * rand_max_r;
            for (int i = 0; i < n*r; i++) H[i] = rand() * rand_max_r;
            for (int i = 0; i < m*n; i++) V[i] = rand() * rand_max_r;
            fact_function(V, W, H, m, n, r, max_iterations, 0.005);
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
    
        free(V);
        free(W);
        free(H);


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
        printf("./build/performance ");
        printf("<program number [1,2,3,4,5]> ");
        printf("<min size matrix> ");
        printf("<max size matrix> ");
        printf("<number of test>");
        printf("<output_file>[?]\n\n");

        printf("Program index:\n");
        printf("\t1. Baseline 1\n");
        printf("\t2. Baseline 2\n");
        printf("\t3. Optimisation 0\n");
        printf("\t4. Optimisation 1\n");
        printf("\t5. Alg Opt 1\n");
        printf("\t6. Alg Opt 2\n");
        printf("\t7. Optimisation 2\n");
        printf("\t8. Optimisation 3\n");
        printf("\t9. Optimisation 21\n");
        printf("\t10. Optimisation 22\n");
        printf("\t11. Optimisation 23\n");
        printf("\t12. Optimisation 24\n");
        printf("\t13. Optimisation 31\n");
        printf("\t14. Optimisation 32\n");
        printf("\t15. Optimisation 33\n");
        printf("\t16. Optimisation 34\n");
        printf("\t17. Optimisation 35\n");
        printf("\t18. Optimisation 36\n");
        printf("\t19. Optimisation 41\n");
        printf("\t20. Optimisation 46\n");
        printf("\t21. Optimisation 47\n");
        printf("\t22. Optimisation 45\n");



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

    case 3:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt0, &nnm_cost_2);
        break;
    
    case 4:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt1 ,&nnm_cost_2);
        break;

    case 5:
        optimization(tests, min, max, b, fout, &nnm_factorization_aopt1 ,&nnm_cost_2);
        break;

    case 6:
        optimization(tests, min, max, b, fout, &nnm_factorization_aopt2 ,&nnm_cost_2);
        break;

    case 7:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt2 ,&nnm_cost_2);
        break;

    case 8:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt3 ,&nnm_cost_2);
        break;
    
    case 9:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt21 ,&nnm_cost_2);
        break;
    
    case 10:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt22 ,&nnm_cost_2);
        break;
    
    case 11:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt23 ,&nnm_cost_2);
        break;

    case 12:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt24 ,&nnm_cost_2);
        break;

    case 13:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt31, &nnm_cost_2);
        break;

    case 14:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt32, &nnm_cost_2);
        break;

    case 15:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt33, &nnm_cost_2);
        break;

    case 16:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt34, &nnm_cost_2);
        break;

    case 17:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt35, &nnm_cost_2);
        break;

    case 18:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt36, &nnm_cost_2);
        break;

    case 19:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt41, &nnm_cost_2);
        break;

    case 20:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt46, &nnm_cost_2);
        break;

    case 21:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt47, &nnm_cost_2);
        break;

    case 22:
        optimization(tests, min, max, b, fout, &nnm_factorization_opt45, &nnm_cost_2);
        break;


    default:
        break;
    }
    if( fout != NULL)
        fclose(fout);

    return 0;
}



