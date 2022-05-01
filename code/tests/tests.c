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

int test_matrix_mult(void (*mmul) (Matrix *A, Matrix *B, Matrix *R)){
    Matrix A; 
    Matrix B;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("A_mul.matrix", "r");
    if (f == NULL){
        printf("Error opening file\n");
        return 0;
    }
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("B_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("R_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    mmul(&A, &B, &R_Computed);

    for(int i=0; i<R_Computed.n_col*R_Computed.n_row; i++){
        if( fabs(R_Computed.M[i] - R_Computed.M[i]) > TOLERANCE){
            matrix_deallocation(&A);
            matrix_deallocation(&B);
            matrix_deallocation(&R_Real);
            return -1;
        }
    }

    matrix_deallocation(&A);
    matrix_deallocation(&B);
    matrix_deallocation(&R_Real);

    return 0;

}

int test_matrix_ltrans_mult(void (*mmulltrans) (Matrix *A, Matrix *B, Matrix *R)) {
    Matrix A; 
    Matrix B;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("A_ltrans_mul.matrix", "r");
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("B_ltrans_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("R_ltrans_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    mmulltrans(&A, &B, &R_Computed);

    for(int i=0; i<R_Computed.n_col*R_Computed.n_row; i++){
        if( fabs(R_Computed.M[i] - R_Computed.M[i]) > TOLERANCE){
            matrix_deallocation(&A);
            matrix_deallocation(&B);
            matrix_deallocation(&R_Real);
            return -1;
        }
    }

    matrix_deallocation(&A);
    matrix_deallocation(&B);
    matrix_deallocation(&R_Real);

    return 0;

}

int test_matrix_rtrans_mult(void (*mmulrtrans) (Matrix *A, Matrix *B, Matrix *R)){
    Matrix A; 
    Matrix B;
    Matrix R_Real;
    Matrix R_Computed;

    FILE *f;

    f = fopen("A_rtrans_mul.matrix", "r");
    read_matrix_from_file(&A, f);
    fclose(f);

    f = fopen("B_rtrans_mul.matrix", "r");
    read_matrix_from_file(&B, f);
    fclose(f);

    f = fopen("R_rtrans_mul.matrix", "r");
    read_matrix_from_file(&R_Real, f);
    fclose(f);

    matrix_allocation(&R_Computed, R_Real.n_row, R_Real.n_col);
    mmulrtrans(&A, &B, &R_Computed);

    for(int i=0; i<R_Computed.n_col*R_Computed.n_row; i++){
        if( fabs(R_Computed.M[i] - R_Computed.M[i]) > TOLERANCE){
            matrix_deallocation(&A);
            matrix_deallocation(&B);
            matrix_deallocation(&R_Real);
            return -1;
        }
    }

    matrix_deallocation(&A);
    matrix_deallocation(&B);
    matrix_deallocation(&R_Real);
    
    return 0;
}

int test_nnm(double (*nnm) (Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon)){
    
    double resultBS1, resultBS2;
    srand(SEED);
    int m = 400;
    int n = 400;
    int r = 20;
    int min = 0;
    int max = 100;
    int maxIteration = 1000;
    int epsilon = 0.05;
    Matrices matrices;
    Matrix W_temp;
    Matrix H_temp;

    allocate_base_matrices(&matrices, m, n, r);
    
    random_matrix_init(&matrices.V,min, max);
    random_matrix_init(&matrices.W,min, max);

    // random_acol_matrix_init(&b->bs1Matrices.V,&b->bs1Matrices.W, 3);
    // random_acol_matrix_init(&b->bs2Matrices.V,&b->bs2Matrices.W, 3);

    random_matrix_init(&matrices.H,min, max);
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

    resultBS2 = nnm(&matrices.V, &W_temp,
                                      &H_temp, maxIteration, epsilon);
    if (fabs(resultBS1 - resultBS2) > 0.000001) {
        printf("Results: error_bs1=%lf, error_implementation=%lf\t", resultBS1, resultBS2);
        return -1;
    }
   
    for(int i=0; i<matrices.H.n_col*matrices.H.n_row;i++){
        if (fabs(H_temp.M[i] - matrices.H.M[i]) > 0.000001){
            printf("H_bs1[%d][%d] - H_implementation[%d][%d] diff by %lf\t", i,i/H_temp.n_col,i,i/H_temp.n_col,fabs(H_temp.M[i] - matrices.H.M[i]));
            return -1;
        }
    }

    for(int i=0; i<matrices.W.n_col*matrices.W.n_row;i++){
        if (fabs(W_temp.M[i] - matrices.W.M[i]) > 0.000001){
            printf("W_bs1[%d][%d] - W_implementation[%d][%d] diff by %lf\t", i,i/W_temp.n_col,i,i/W_temp.n_col,fabs(W_temp.M[i] - matrices.W.M[i]));
            return -1;
        }
    }
    return 0;

}

int main(int argc, char const *argv[])
{
    char *default_path = "/home/asl/asl-2022/code/tests/inputs";
    if(argc == 1){
        printf("No path specified, using default: %s \n", default_path);
        chdir(default_path);
    }else{
        chdir(argv[1]);
    }

    // Please add following the number of functions you would like to test:
    int n = 2;
    void (*mmulrtrans[n]) (Matrix *A, Matrix *B, Matrix *R);
    void (*mmulltrans[n]) (Matrix *A, Matrix *B, Matrix *R);
    void (*mmul[n]) (Matrix *A, Matrix *B, Matrix *R);
    double (*nnm[n]) (Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon);

    // Now add the functions you would like to test from the impleemntations:
    mmulltrans[0] = matrix_ltrans_mul_bs1;
    mmulltrans[1] = matrix_ltrans_mul_bs2;

    mmulrtrans[0] = matrix_rtrans_mul_bs1;
    mmulrtrans[1] = matrix_rtrans_mul_bs2;

    mmul[0] = matrix_rtrans_mul_bs1;
    mmul[1] = matrix_rtrans_mul_bs2;

    nnm[0] = nnm_factorization_bs1;
    nnm[1] = nnm_factorization_bs2;
    
    printf("################ Starting general test ################\n\n");

    int result;
    int sum_results = 0;

    for (int i = 0; i < n; i++) {
            printf("Matrix mult implementation %i:", i);
            result = test_matrix_mult(mmul[i]);
            print_test_status(result);
            sum_results += result;

            printf("Matrix ltrans mult implementation %i:", i);
            result = test_matrix_ltrans_mult(mmulltrans[i]);
            print_test_status(result);
            sum_results += result;

            printf("Matrix rtrans mult implementation %i:", i);
            result = test_matrix_rtrans_mult(mmulrtrans[i]);
            print_test_status(result);
            sum_results += result;

            printf("NNM implementation %i compared to bs1:\t", i);
            result = test_nnm(nnm[i]);
            print_test_status(result);
            sum_results += result;
    }
    
    if(sum_results == 0){
        printf("\nTest completed. All test \e[32mPASSED\e[0m\n");
    }else{
        printf("\nTest completed. Numer of test \e[0;31mFAILED\e[0m: %d\n", sum_results*-1);
    }
    return 0;
}
