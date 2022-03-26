
//#error Please comment out the next two lines under linux, then comment this error
//#include "stdafx.h"  //Visual studio expects this line to be the first one, comment out if different compiler
//#include <windows.h> // Include if under windows

#include "../baseline1/baseline1.h"
#include "../baseline2/baseline2.h"

#ifndef WIN32
#include <sys/time.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifdef __x86_64__
#include "tsc_x86.h"
#endif

#define NUM_RUNS 10
#define CYCLES_REQUIRED 1e8
#define FREQUENCY 3.5e9	
#define CALIBRATE


void baseline1_old( int numTests, int min, int max){
    Matrix V;
    Matrix W, H;
    int m, n, r;

    srand(time(NULL));

    long cost;
    double cycles, performance;
    clock_t start, end;
    int num_runs = NUM_RUNS;

    int steps = (max - min) / numTests;
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
        printf("--- cost(flops):%ld, performance(flops/cycle):%f\n\n", cost, performance);

        matrix_deallocation(&V);
        matrix_deallocation(&W);
        matrix_deallocation(&H);
    }
}

void baseline2( int numTests, int min, int max){
    vMatrix V;
    vMatrix W, H;
    myInt64 m, n, r;
    printf("Baseline 2 performance evaluation\n\n");
    srand(time(NULL));

    myInt64 cost;
    double performance;
    
    int num_runs = NUM_RUNS;

    int steps = (max - min) / numTests;
    for (int i = min; i < max; i += steps) {
        m = i;
        n = i + steps;
        r = 12;

        V.n_row = m;
        V.n_col = n;
        v_matrix_allocation(&V);

        W.n_row = m;
        W.n_col = r;
        v_matrix_allocation(&W);

        H.n_row = r;
        H.n_col = n;
        v_matrix_allocation(&H);

        //Call adequate cost functions
        //(3 + 7*r*n + 3*m*r + 6n*m + 10*m*r*n + 2*m*r*r + 2*r*r*n) * Number_of_iterations
        cost = 0;
        // init of W cost
        cost += 4 * m * r;
        // init of H cost
        cost += 4 * r * n;
        // nnmf cost
        cost += (3 + 7*r*n + 3*m*r + 6*n*m + 10*m*r*n + 2*m*r*r + 2*r*r*n) * 100;
        

        #ifdef __x86_64__
        myInt64 cycles;
        myInt64 start;
        #ifdef CALIBRATE
        while(num_runs < (1 << 14)) {
            start = start_tsc();
            for (int j = 0; j < num_runs; j++) {

                random_v_matrix_init(&W, 0, 1);
                random_v_matrix_init(&H, 0, 1);
                random_v_matrix_init(&V, 0, 1);

                nnm_factorization_bs2(&V, &W, &H, 100, 0.005);
            }
            cycles = stop_tsc(start);

            if(cycles >= CYCLES_REQUIRED) break;

            num_runs *= 2;
        }
        #endif

        start = start_tsc();
        random_v_matrix_init(&V, 0, 1);
        for (int j = 0; j < num_runs; j++) {

            random_v_matrix_init(&W, 0, 1);
            random_v_matrix_init(&H, 0, 1);
            

            nnm_factorization_bs2(&V, &W, &H, 100, 0.005);
        }

        cycles = stop_tsc(start)/num_runs;
        performance =  (double)cost / (double)cycles;

        #endif

        #ifndef __x86_64__
        double cycles;
        clock_t start, end;
        #ifdef CALIBRATE
        while(num_runs < (1 << 14)) {
            start = clock();
            for (int j = 0; j < num_runs; j++) {

                random_v_matrix_init(&W, 0, 1);
                random_v_matrix_init(&H, 0, 1);
                random_v_matrix_init(&V, 0, 1);

                nnm_factorization_bs2(&V, &W, &H, 100, 0.005);
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

            random_v_matrix_init(&W, 0, 1);
            random_v_matrix_init(&H, 0, 1);
            random_v_matrix_init(&V, 0, 1);

            nnm_factorization_bs2(&V, &W, &H, 100, 0.005);
        }
        end = clock();

        cycles = (double)(end - start) / num_runs;
        //cycles = (cycles / CLOCKS_PER_SEC) * FREQUENCY;
        //performance = cost / cycles;
        performance = ((cost / cycles) / FREQUENCY) * CLOCKS_PER_SEC;
        

        #endif

        v_matrix_deallocation(&V);
        v_matrix_deallocation(&W);
        v_matrix_deallocation(&H);

        printf("Sizes: m=%llu, n=%llu, r=%llu:\n", m, n, r);
        printf("--- cost(flops):%llu, cycles:%llu, performance(flops/cycle):%lf\n\n", cost, cycles, performance);


    }
}

void baseline1( int numTests, int min, int max){
    Matrix V;
    Matrix W, H;
    myInt64 m, n, r;
    printf("Baseline 1 performance evaluation\n\n");
    srand(time(NULL));

    myInt64 cost;
    double performance;
    
    int num_runs = NUM_RUNS;

    int steps = (max - min) / numTests;
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
        //(3 + 7*r*n + 3*m*r + 6n*m + 10*m*r*n + 2*m*r*r + 2*r*r*n) * Number_of_iterations
        //Call adequate cost functions

        cost = random_matrix_init_cost(&W);
        cost += random_matrix_init_cost(&H);
        cost += random_matrix_init_cost(&V); // should be removed since the V matrix is given

        cost += nnm_factorization_bs1_cost(&V, &W, &H, 100);
        

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

                nnm_factorization_bs1(&V, &W, &H, 100, 0.005);
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

            nnm_factorization_bs1(&V, &W, &H, 100, 0.005);
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
        printf("<number of test>\n\n");

        return 0;
    }

    int b, tests, min, max;
    b = atoi(argv[1]);
    
    min = atoi(argv[2]);
    max = atoi(argv[3]);
    tests = atoi(argv[4]);
    
    switch(b){
    case 1:
        baseline1(tests, min, max);
        break;
    
    case 2:
        baseline2(tests, min, max);
        break;
    
    default:
        break;
    }

    
    

    return 0;
}



