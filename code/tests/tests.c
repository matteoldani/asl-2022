#include <tests/tests.h>
#define TOLERANCE 0

static void print_test_status(int return_value){
    if(return_value == -1){
        printf("\t\e[0;31mFAIL\e[0m\n");
    }else{
        printf("\t\e[32mSUCESS\e[0m\n");
    }
}

static void read_matrix_from_file(Matrix *M, FILE *f){
    int n_row;
    int n_col;
    
    fscanf(f, "%d", &n_row);
    fscanf(f, "%d", &n_col);

    matrix_allocation(M, n_row, n_col);
    
    for (int i=0; i<M->n_row * M->n_col; i++){
        fscanf(f, "%lf", &(M->M[i]));
    }
}


int test_matrix_mult_bs1(){
    Matrix A; 
    Matrix B;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("inputs/A_mul.matrix", "r");
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("inputs/B_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("inputs/R_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    matrix_mul_bs1(&A, &B, &R_Computed);


    for(int i=0; i<R_Computed.n_col*R_Computed.n_row; i++){
        if( fabs(R_Computed.M[i] - R_Computed.M[i]) > TOLERANCE){
            return -1;
        }
    }

    return 0;

}

int test_matrix_ltrans_mult_bs1(){
    Matrix A; 
    Matrix B;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("inputs/A_ltrans_mul.matrix", "r");
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("inputs/B_ltrans_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("inputs/R_ltrans_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    matrix_ltrans_mul_bs1(&A, &B, &R_Computed);

    for(int i=0; i<R_Computed.n_col*R_Computed.n_row; i++){
        if( fabs(R_Computed.M[i] - R_Computed.M[i]) > TOLERANCE){
            return -1;
        }
    }

    return 0;
}

int test_matrix_rtrans_mult_bs1(){
    Matrix A; 
    Matrix B;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("inputs/A_rtrans_mul.matrix", "r");
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("inputs/B_rtrans_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("inputs/R_rtrans_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    matrix_rtrans_mul_bs1(&A, &B, &R_Computed);

    for(int i=0; i<R_Computed.n_col*R_Computed.n_row; i++){
        if( fabs(R_Computed.M[i] - R_Computed.M[i]) > TOLERANCE){
            return -1;
        }
    }

    return 0;
}

int test_matrix_mult_bs2(){
    Matrix A; 
    Matrix B;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("inputs/A_mul.matrix", "r");
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("inputs/B_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("inputs/R_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    matrix_mul_bs2(&A, &B, &R_Computed);

    for(int i=0; i<R_Computed.n_col*R_Computed.n_row; i++){
        if( fabs(R_Computed.M[i] - R_Computed.M[i]) > TOLERANCE){
            return -1;
        }
    }

    return 0;
}

int test_matrix_ltrans_mult_bs2(){
    Matrix A; 
    Matrix B;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("inputs/A_ltrans_mul.matrix", "r");
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("inputs/B_ltrans_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("inputs/R_ltrans_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    matrix_ltrans_mul_bs2(&A, &B, &R_Computed);

    for(int i=0; i<R_Computed.n_col*R_Computed.n_row; i++){
        if( fabs(R_Computed.M[i] - R_Computed.M[i]) > TOLERANCE){
            return -1;
        }
    }

    return 0;
}

int test_matrix_rtrans_mult_bs2(){
    Matrix A; 
    Matrix B ;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("inputs/A_rtrans_mul.matrix", "r");
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("inputs/B_rtrans_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("inputs/R_rtrans_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    matrix_rtrans_mul_bs2(&A, &B, &R_Computed);

    for(int i=0; i<R_Computed.n_col*R_Computed.n_row; i++){
        if( fabs(R_Computed.M[i] - R_Computed.M[i]) > TOLERANCE){
            return -1;
        }
    }

    return 0;
}

int test_nnm_bs1(){
    Matrix W_computed; 
    Matrix H_computed;
    Matrix V;
    Matrix W_real;
    Matrix H_real;


    FILE *f;

    f = fopen("inputs/W_nnm_init.matrix", "r");
    read_matrix_from_file(&W_computed, f);
    fclose(f);


    f = fopen("inputs/H_nnm_init.matrix", "r");
    read_matrix_from_file(&H_computed, f);
    fclose(f);

    f = fopen("inputs/V_nnm.matrix", "r");
    read_matrix_from_file(&V, f);
    fclose(f);

    f = fopen("inputs/W_nnm.matrix", "r");
    read_matrix_from_file(&W_real, f);
    fclose(f);

    f = fopen("inputs/H_nnm.matrix", "r");
    read_matrix_from_file(&H_real, f);
    fclose(f);


    nnm_factorization_bs1(&V, &W_computed, &H_computed, 1000, 0.001);

    for(int i=0; i<H_computed.n_col*H_computed.n_row; i++){
        if( fabs(H_computed.M[i] - H_real.M[i]) > TOLERANCE){
            printf("Error: %lf\t",fabs(H_computed.M[i] - H_real.M[i]) );
            return -1;
        }
    }

    return 0;


}


int test_nnm_bs2(){
    Matrix W_computed; 
    Matrix H_computed;
    Matrix V;
    Matrix W_real;
    Matrix H_real;


    FILE *f;

    f = fopen("inputs/W_nnm_init.matrix", "r");
    read_matrix_from_file(&W_computed, f);
    fclose(f);


    f = fopen("inputs/H_nnm_init.matrix", "r");
    read_matrix_from_file(&H_computed, f);
    fclose(f);

    f = fopen("inputs/V_nnm.matrix", "r");
    read_matrix_from_file(&V, f);
    fclose(f);

    f = fopen("inputs/W_nnm.matrix", "r");
    read_matrix_from_file(&W_real, f);
    fclose(f);

    f = fopen("inputs/H_nnm.matrix", "r");
    read_matrix_from_file(&H_real, f);
    fclose(f);


    nnm_factorization_bs2(&V, &W_computed, &H_computed, 1000, 0.001);

    for(int i=0; i<H_computed.n_col*H_computed.n_row; i++){
        if( fabs(H_computed.M[i] - H_real.M[i]) > TOLERANCE){
            printf("Error: %lf\t",fabs(H_computed.M[i] - H_real.M[i]) );
            return -1;
        }
    }

    return 0;


}


int main(int argc, char const *argv[])
{
    if(argc == 1){
        printf("Pleas specify the full path where to find the input for the tests\n");
    }
    chdir(argv[1]);
    printf("################ Startinf general test ################\n\n");

    int result;
    int sum_results = 0;

    printf("Matrix mult bs1:");
    result = test_matrix_mult_bs1();
    print_test_status(result);
    sum_results += result;

    printf("Matrix mult bs1:");
    result = test_matrix_mult_bs1();
    print_test_status(result);
    sum_results += result;

    printf("Matrix ltrans mult bs1:");
    result = test_matrix_ltrans_mult_bs1();
    print_test_status(result);
    sum_results += result;

    printf("Matrix rtrans mult bs1:");
    result = test_matrix_rtrans_mult_bs1();
    print_test_status(result);
    sum_results += result;

    printf("Matrix mult bs2:");
    result = test_matrix_mult_bs2();
    print_test_status(result);
    sum_results += result;

    printf("Matrix ltrans mult bs2:");
    result = test_matrix_ltrans_mult_bs2();
    print_test_status(result);
    sum_results += result;

    printf("Matrix rtrans mult bs2:");
    result = test_matrix_rtrans_mult_bs2();
    print_test_status(result);
    sum_results += result;

    printf("NNM bs1:");
    result = test_nnm_bs1();
    print_test_status(result);
    sum_results += result;

    printf("NNM bs2:");
    result = test_nnm_bs2();
    print_test_status(result);
    sum_results += result;

    if(sum_results == 0){
        printf("Test completed. All test PASSED\n");
    }else{
        printf("Test completed. Numer of test failed: %d\n", sum_results*-1);
    }
    return 0;
}
