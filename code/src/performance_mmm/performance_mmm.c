#include <mmm/mmm_0.h>
#include <mmm/mmm_1.h>
#include <mmm/mmm_2.h>
#include <mmm/mmm_3.h>
#include <stdlib.h>


#include <asl.h>

#include <performance/performance.h>
#include <baselines/baselines_utils.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define CPU_FREQ 3.5e9
typedef void (*mmm_m) (Matrix *, Matrix *, Matrix *);
typedef void (*mmm_v) (double *, int , int , double *, int , int , double *, int , int);

mmm_m run_mmm_m;
mmm_v run_mmm_v;

static int double_size = sizeof(double);

void transpose(double *src, double *dst,  const int N, const int M) {

    int nB = 1;
    int nBM = nB * M;
    int src_i = 0, src_ii;

    for(int i = 0; i < N; i += nB) {
        for(int j = 0; j < M; j += nB) {
            src_ii = src_i;
            for(int ii = i; ii < i + nB; ii++) {
                for(int jj = j; jj < j + nB; jj++)
                    dst[N*jj + ii] = src[src_ii + jj];
                src_ii += M;
            }
        }
        src_i += nBM;
    }   
}

void pad_matrix(double ** M, int *r, int *c){
    int temp_r;
    int temp_c;

    if( (*r) %BLOCK_SIZE_MMUL != 0){
        temp_r = (((*r) / BLOCK_SIZE_MMUL ) + 1)*BLOCK_SIZE_MMUL;   
    }else{
        temp_r = *r;
    }

    if((*c)%BLOCK_SIZE_MMUL != 0){
        temp_c = (((*c )/ BLOCK_SIZE_MMUL) + 1) * BLOCK_SIZE_MMUL;
    }else{
        temp_c = *c;
    }

    double *new_Mt;

    *M = realloc(*M, double_size * (*c) * temp_r);
    // i need to pad the rows before and the cols after transposing
    memset(&(*M)[(*c)*(*r)], 1.5, double_size * (temp_r-(*r)) * (*c));

    new_Mt = malloc(double_size * temp_c * temp_r);
    transpose(*M, new_Mt, temp_r, *c);
    memset(&new_Mt[temp_r * (*c)], 2.5, double_size * (temp_c - (*c)) * temp_r);

    free(*M);
    *M = malloc(double_size * temp_c * temp_r);
    *c = temp_c;
    *r = temp_r;
    transpose(new_Mt, *M, temp_c, temp_r); 

    free(new_Mt);


}

void unpad_matrix(double **M, int *r, int *c, int original_r, int original_c){

    // lets suppose that are always row majour

    // i can remove the last useless rows
    *M = realloc(*M, (*c) * original_r * double_size);

    // i need to transpose and remove the rest
    double *new_Mt = malloc((*c) * original_r * double_size );
    transpose(*M, new_Mt, original_r, *c);

    // i need to resize the transoposed
    new_Mt = realloc(new_Mt, double_size * original_c * original_r);

    // ie need to transpose back
    free(*M);
    *M = malloc(double_size * original_c * original_r);
    transpose(new_Mt, *M, original_c, original_r);

    *r = original_r;
    *c = original_c;

    free(new_Mt);
    
}

void perf_m(int numTests, int min, int max, int opt, FILE * fout){
    Matrix V;
    Matrix W, H;
    int m, n, r;
    printf("MMM %d performance evaluation\n\n", opt);
    srand(SEED);

    myInt64 cost = 0;
    double performance;
    
    int num_runs = NUM_RUNS;
    double rand_max_r = 1 / (double)RAND_MAX;

    int steps = (max - min) / numTests;
    printf("Steps %d\n", steps);
    for (int i = min; i < max; i += steps) {  

        m = i;
        n = i; //+ steps; 
        r = RANK;

        cost = 2 * m * r * n;

        matrix_allocation(&V, m, n);
        matrix_allocation(&W, m, r);
        matrix_allocation(&H, r, n);

        random_matrix_init(&W, 0, 1);
        random_matrix_init(&H, 0, 1);
        random_matrix_init(&V, 0, 1);

        #ifdef __x86_64__
        myInt64 cycles;
        myInt64 start;
        #ifdef CALIBRATE
        while(num_runs < (1 << 14)) {
            start = start_tsc();
            for (int j = 0; j < num_runs; j++) {
                 run_mmm_m(&W, &H, &V);
            }
            cycles = stop_tsc(start);

            if(cycles >= CYCLES_REQUIRED) break;

            num_runs *= 2;
        }
        #endif 

        start = start_tsc();

        for (int j = 0; j < num_runs; j++) {
            run_mmm_m(&W, &H, &V);
        }

        cycles = stop_tsc(start)/num_runs;
        performance =  (double )cost / (double) cycles;

        #endif

        printf("Sizes: m=%llu, n=%llu, r=%llu:\n", m, n, r);
        printf("--- cost(flops):%llu, cycles:%llu, performance(flops/cycle):%lf\n\n", cost, cycles, performance);
        
         if(fout != NULL){
            fprintf(fout, "%llu,%llu,%llu,%llu,%lf, %llu\n", m, r, n, cost, performance, cycles);
        }

        matrix_deallocation(&V);
        matrix_deallocation(&W);
        matrix_deallocation(&H);
    }
}


void perf_v(int numTests, int min, int max, int opt, FILE * fout){
    int m, n, r;
    printf("MMM %d performance evaluation\n\n", opt);
    srand(SEED);

    myInt64 cost = 0;
    double performance;
    
    int num_runs = NUM_RUNS;
    double rand_max_r = 1 / (double)RAND_MAX;

    int steps = (max - min) / numTests;
    printf("Steps %d\n", steps);
    for (int i = min; i < max; i += steps) {
    //for (int i = max - steps; i >= min; i -= steps) {
    
        m = i;
        n = i; //+ steps; 
        r = RANK;
   
        double * A, * B, * C;

        A = malloc(sizeof(double) * m * r);
        B = malloc(sizeof(double) * r * n);
        C = malloc(sizeof(double) * m * n);
    

        for(int i = 0; i<m*r; i++){
            A[i]= rand_from(0, 1);
        }

        for(int i = 0; i<n*r; i++){
            B[i]= rand_from(0, 1);
        }
        int original_m = m;
        int original_r = r;
        int original_n = n;


        int temp_m = m;
        int temp_n = n;
        int temp_r = r;
        if(run_mmm_v == &matrix_mul_3){
            pad_matrix(&A, &temp_m, &temp_r);
            pad_matrix(&B, &r, &temp_n);
            pad_matrix(&C, &m, &n);
        }

        cost = 2 * m * r * n;

        double * A_p = aligned_alloc(32, sizeof(double) * m * r);
        double * B_p = aligned_alloc(32, sizeof(double) * r * n);
        double * C_p = aligned_alloc(32, sizeof(double) * m * n);
        
        memcpy(A_p, A, m * r * double_size);
        memcpy(B_p, B, r * n * double_size);
        memcpy(C_p, C, m * n * double_size);

   
        // printf("Pointer A: %p\n", A);

        // if (!((int)((const void *)(A_p)) % (32) == 0)){
        //   //TODO handle alignment 
        //   printf("A NOT aligned\n");
        // }
        // if (!((int)((const void *)(B_p)) % (32) == 0)){
        //   //TODO handle alignment 
        //   printf("B NOT aligned\n");
        // }
        // if (!((int)((const void *)(C_p)) % (32) == 0)){
        //   //TODO handle alignment 
        //   printf("C NOT aligned\n");
        // }


        //Call adequate cost functio
   


        #ifdef __x86_64__
        myInt64 cycles;
        myInt64 start;
        #ifdef CALIBRATE
        while(num_runs < (1 << 14)) {
            start = start_tsc();
            for (int j = 0; j < num_runs; j++) {
                 run_mmm_v(A_p, m, r, B_p, r, n, C_p, m, n);
            }
            cycles = stop_tsc(start);

            if(cycles >= CYCLES_REQUIRED) break;

            num_runs *= 2;
        }
        #endif 

        start = start_tsc();

        for (int j = 0; j < num_runs; j++) {
                run_mmm_v(A_p, m, r, B_p, r, n, C_p, m, n);
        }

        cycles = stop_tsc(start)/num_runs;
        performance =  (double )cost / (double) cycles;

        #endif

        printf("Sizes: m=%llu, n=%llu, r=%llu:\n", m, n, r);
        printf("--- cost(flops):%llu, cycles:%llu, performance(flops/cycle):%lf\n\n", cost, cycles, performance);
        
         if(fout != NULL){
            fprintf(fout, "%llu,%llu,%llu,%llu,%lf, %llu\n", original_m, original_r, original_n, cost, performance, cycles);
        }

        free(A);
        free(B);
        free(C);

        free(A_p);
        free(B_p);
        free(C_p);
    }
}


int main(int argc, char const *argv[])
{
    if(argc < 5){
        printf("Usage:\n");
        printf("\t./build/run_mmm <id> <min> <max> <num-test>\n\n");

        printf("IDs:\n");
        printf("\t- 0: Matrix Mul 2\n");
        printf("\t- 1: Matrix Mul 3\n");
        printf("\n");

        return 0;
    }

    
    int tests, min, max, id;
    FILE *fout = NULL;
    id = atoi(argv[1]);
    min = atoi(argv[2]);
    max = atoi(argv[3]);
    tests = atoi(argv[4]);


    if (argc > 5){
        fout = fopen(argv[5], "w+");
        if (fout == NULL){
            printf("Can't open output file\n");
            exit(-1);
        }
        fprintf(fout, "m,r,n,cost,performance,cycles\n");
    }

    fflush(stdout);


    fflush(stdout); 
    switch (id)
    {

    case 0:
        run_mmm_m = &matrix_mul_0;
        perf_m(tests, min, max, id, fout);
        break;
    case 1:
        run_mmm_m = &matrix_mul_1;
        perf_m(tests, min, max, id, fout);
        break;
    case 2:
        run_mmm_v = &matrix_mul_2;
        perf_v(tests, min, max, id, fout);
        break;
    case 3:
        run_mmm_v = &matrix_mul_3;
        perf_v(tests, min, max, id, fout);
        break;

    default:
        printf("Wrong id\n");
        break;
    }


    

    return 0;
}