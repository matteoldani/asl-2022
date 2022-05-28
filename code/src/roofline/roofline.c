#include <roofline/roofline.h>

void baseline(int numTests, int input_size, int b, FILE * fout, fact_function fact_function){
    Matrix V;
    Matrix W, H;
    myInt64 m, n, r;
    srand(SEED);
    
    m = input_size;
    n = input_size;
    r = RANK;

    matrix_allocation(&V, m, n);
    matrix_allocation(&W, m, r);
    matrix_allocation(&H, r, n);

    for (int j = 0; j < numTests; j++) {

        random_matrix_init(&W, 0, 1);
        random_matrix_init(&H, 0, 1);
        random_matrix_init(&V, 0, 1);

        fact_function(&V, &W, &H, 100, 0.005);
    }
    matrix_deallocation(&V);
    matrix_deallocation(&W);
    matrix_deallocation(&H);
}

void optimization(int numTests, int input_size, int opt, FILE * fout, opt_fact_function fact_function){
    double* V;
    double* W;
    double* H;
    myInt64 m, n, r;
    srand(SEED);
    
    m = input_size;
    n = input_size; 
    r = RANK;
    V = malloc(m * n * sizeof(double));
    W = malloc(m * r * sizeof(double));
    H = malloc(r * n * sizeof(double));

    double rand_max_r = 1 / (double)RAND_MAX;

    for (int j = 0; j < numTests; j++) {

        for (int i = 0; i < m*r; i++) W[i] = rand() * rand_max_r;
        for (int i = 0; i < n*r; i++) H[i] = rand() * rand_max_r;
        for (int i = 0; i < m*n; i++) V[i] = rand() * rand_max_r;

        fact_function(V, W, H, m, n, r, 100, 0.005);
    }
    
    free(V);
    free(W);
    free(H);
}


int main(int argc, char const* argv[])
{
    if(argc <= 1){
        printf("How to use this tool:\n");
        printf("./build/performance ");
        printf("<program number [1,2,3,4,5]> ");
        printf("<input size> ");
        printf("<number of test>\n");

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


        return 0;
    }

    int b, tests, input_size;
    FILE *fout = NULL;
    b = atoi(argv[1]);
    
    input_size = atoi(argv[2]);
    tests = atoi(argv[3]);

    switch(b){
    case 1:
        baseline(tests, input_size, b, fout, &nnm_factorization_bs1);
        break;
    
    case 2:
        baseline(tests, input_size, b, fout, &nnm_factorization_bs2);
        break;

    case 3:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt0);
        break;
    
    case 4:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt1);
        break;

    case 5:
        optimization(tests, input_size, b, fout, &nnm_factorization_aopt1);
        break;

    case 6:
        optimization(tests, input_size, b, fout, &nnm_factorization_aopt2);
        break;

    case 7:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt2);
        break;

    case 8:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt3);
        break;
    
    case 9:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt21);
        break;
    
    case 10:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt22);
        break;
    
    case 11:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt23);
        break;

    case 12:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt24);
        break;

    case 13:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt31);
        break;

    case 14:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt32);
        break;

    case 15:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt33);
        break;

    case 16:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt34);
        break;

    default:
        break;
    }
    if( fout != NULL)
        fclose(fout);

    return 0;
}
