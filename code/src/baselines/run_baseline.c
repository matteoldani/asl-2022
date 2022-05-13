#include <baselines/baseline1.h>
#include <baselines/baseline2.h>

/**
 * @brief represents a dynamic allocated matrix
 * @param M     is the matrix
 * @param n_row is the number of rows
 * @param n_col is the number of cols
 */

typedef double (*fact_function) (Matrix *, Matrix *, Matrix *, int, double);
int main(int argc, char const *argv[]) {

    Matrix V;
    Matrix W, H;

    fact_function run_factorization;
    int m, n, r, b = 0;
    m = 1000;
    n = 1000;
    r = 12;
    if (argc != 3 && argc != 5){
        printf("This program can be run in tow different modes:\n");
        printf("\t./baseline <baseline number> <m> <n> <r>\n");
        printf("\t./baseline <baseline number> <file-path>\n");
        return -1;
    }
    FILE *fp = NULL;
    if (argc == 5){
        b = atoi(argv[1]);
        m = atoi(argv[2]); 
        n = atoi(argv[3]);
        r = atoi(argv[4]);

        srand(SEED);

        matrix_allocation(&V, m, n);
        random_matrix_init(&V, 0, 1); 
    }

    if(argc == 3){
        fp = fopen(argv[2], "r");
        if(fp == NULL){
            printf("File not found, exiting\n");
            return -1;
        }

        allocate_from_file(&V, &r, fp);
        m = V.n_row;
        n = V.n_col;
    }

    if(b == 1){
        run_factorization = &nnm_factorization_bs1;
    }else{
        run_factorization = &nnm_factorization_bs2;
    }
    


    matrix_allocation(&W, m, r);
    matrix_allocation(&H, r, n);

    random_matrix_init(&W, 0, 1);          // 5 * W->n_row * W->n_col
    random_matrix_init(&H, 0, 1);          // 5 * H->n_row * H->n_col
     

    double err = run_factorization(&V, &W, &H, MAX_ITERATIONS, EPSILON);  // go to baseline1.c nnm_factorization_bs1 for cost
    printf("Error: %lf\n", err);

    matrix_deallocation(&V);
    matrix_deallocation(&W);
    matrix_deallocation(&H);

    return 0;
}