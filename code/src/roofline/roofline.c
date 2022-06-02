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

    case 17:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt35);
        break;

    case 18:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt36);
        break;

    case 19:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt41);
        break;

    case 20:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt42);
        break;

    case 21:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt43);
        break;

    case 22:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt44);
        break;

    case 23:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt45);
        break;

    case 24:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt46);
        break;

    case 25:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt47);
        break;

    case 26:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt37);
        break;

    case 27:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt51);
        break;

    case 28:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt52);
        break;

    case 29:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt53);
        break;
        
    case 30:
        optimization(tests, input_size, b, fout, &nnm_factorization_opt54);
        break;

    default:
        break;
    }
    if( fout != NULL)
        fclose(fout);

    return 0;
}
