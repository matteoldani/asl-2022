/*
   1000, 1000, 10
    Opt 1:

            168’486’041’349      cpu-cycles                                                    (80.00%)
                960’028’549      cache-misses                                                  (80.00%)
                    25’367’071      LLC-loads                                                     (80.00%)
                    12’031’535      LLC-load-misses           #   47.43% of all LL-cache accesses  (80.00%)
            103’447’338’103      FP_ARITH_INST_RETIRED.SCALAR_DOUBLE                                     (80.00%)

                93.964576381 seconds time elapsed

                93.943614000 seconds user
                0.015999000 seconds sys
    baseline 1:
            205’657’641’834      cpu-cycles                                                    (80.00%)
                1’867’038’669      cache-misses                                                  (80.00%)
                3’381’663’745      LLC-loads                                                     (80.00%)
                365’389’096      LLC-load-misses           #   10.81% of all LL-cache accesses  (80.00%)
            105’443’790’083      FP_ARITH_INST_RETIRED.SCALAR_DOUBLE                                     (80.00%)

                114.713733935 seconds time elapsed

                114.686193000 seconds user
                0.019998000 seconds sys
*/



#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>

typedef unsigned long long myInt64;

static unsigned int double_size = sizeof(double);

static void transpose(double *src, double *dst,  const int N, const int M) {
    //#pragma omp parallel for
    //double *dst = malloc(M * N * sizeof(double));
    for(int n = 0; n<N*M; n++) {
        int i = n/N;
        int j = n%N;
        dst[n] = src[M*j + i];
    }   
}

/**
 * @brief compute the multiplication of A and B
 * @param A         is the first factor
 * @param A_n_row   is the number of rows in matrix A
 * @param A_n_col   is the number of columns in matrix A
 * @param B         is the other factor of the multiplication
 * @param B_n_row   is the number of rows in matrix B
 * @param B_n_col   is the number of columns in matrix B
 * @param R         is the matrix that will hold the result
 * @param R_n_row   is the number of rows in the result
 * @param R_n_col   is the number of columns in the result
 */
void matrix_mul_aopt1(double *A, int A_n_row, int A_n_col, double*B, int B_n_row, int B_n_col, double*R, int R_n_row, int R_n_col) {
    int Rij;

    for (int i = 0; i < A_n_row; i++) {
        for (int j = 0; j < B_n_col; j++) {
            Rij = i * R_n_col + j;
            R[Rij] = 0;
            for (int k = 0; k < A_n_col; k++) {
                R[Rij] += A[i * A_n_col + k] * B[k * B_n_col + j];

            }
        }
    }
}

/**
 * @brief compute the multiplication of A^T and B
 * @param A         is the matrix to be transposed
 * @param A_n_row   is the number of rows in matrix A
 * @param A_n_col   is the number of columns in matrix A
 * @param B         is the other factor of the multiplication
 * @param B_n_row   is the number of rows in matrix B
 * @param B_n_col   is the number of columns in matrix B
 * @param R         is the matrix that will hold the result
 * @param R_n_row   is the number of rows in the result
 * @param R_n_col   is the number of columns in the result
 */
void matrix_ltrans_mul_aopt1(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col) {

    int Rij;

    for (int i = 0; i < A_n_col; i++) {
        for (int j = 0; j < B_n_col; j++) {
            Rij = i * R_n_col + j;
            R[Rij] = 0;
            for (int k = 0; k < B_n_row; k++){
                R[Rij] += A[k * A_n_col + i] * B[k * B_n_col + j];


            }
        }
    }
}

/**
 * @brief compute the multiplication of A and B^T
 * @param A         is the other factor of the multiplication
 * @param A_n_row   is the number of rows in matrix A
 * @param A_n_col   is the number of columns in matrix A
 * @param B         is the matrix to be transposed
 * @param B_n_row   is the number of rows in matrix B
 * @param B_n_col   is the number of columns in matrix B
 * @param R         is the matrix that will hold the result
 * @param R_n_row   is the number of rows in the result
 * @param R_n_col   is the number of columns in the result
 */
void matrix_rtrans_mul_aopt1(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col) {
    
    int Rij;

    for (int i = 0; i < A_n_row; i++) {
        for (int j = 0; j < B_n_row; j++) {
            Rij = i * R_n_col + j;
            R[Rij] = 0;
            for (int k = 0; k < A_n_col; k++){
                R[Rij] += A[i * A_n_col + k] * B[j * B_n_col + k];
            }
        }
    }
}

/**
 * @brief computes the error based on the Frobenius norm 0.5*||V-WH||^2. The error is
 *        normalized with the norm V
 *
 * @param approx    is the matrix to store the W*H approximation
 * @param V         is the original matrix
 * @param W         is the first factorization matrix
 * @param H         is the second factorization matrix
 * @param m         is the number of rows in V
 * @param n         is the number of columns in V
 * @param r         is the factorization parameter
 * @param mn        is the number of elements in matrices V and approx
 * @param norm_V    is 1 / the norm of matrix V
 * @return          is the error
 */
inline double error(double* approx, double* V, double* W, double* H, int m, int n, int r, int mn, double norm_V) {

    matrix_mul_aopt1(W, m, r, H, r, n, approx, m, n);

    double norm_approx, temp;

    norm_approx = 0;
    for (int i = 0; i < mn; i++)
    {
        temp = V[i] - approx[i];
        norm_approx += temp * temp;
    }
    norm_approx = sqrt(norm_approx);

    return norm_approx * norm_V;
}

/**
 * @brief computes the non-negative matrix factorisation updating the values stored by the 
 *        factorization functions
 * 
 * @param V             the matrix to be factorized
 * @param W             the first matrix in which V will be factorized
 * @param H             the second matrix in which V will be factorized
 * @param m             the number of rows of V
 * @param n             the number of columns of V
 * @param r             the factorization parameter
 * @param maxIteration  maximum number of iterations that can run
 * @param epsilon       difference between V and W*H that is considered acceptable
 */
double nnm_factorization_aopt1(double *V_rowM, double*W, double*H, int m, int n, int r, int maxIteration, double epsilon) {
    double *Wt, *Ht;
    double *V_colM;
    int rn, rr, mr, mn;
    rn = r * n;
    rr = r * r;
    mr = m * r;
    mn = m * n;
    Wt = malloc(double_size * mr);
    Ht = malloc(double_size * rn);
    V_colM = malloc(double_size * mn);
    
  
    // this is required to be done here to reuse the same run_opt.
    // does not changhe the number of flops
    for (int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
           V_colM[j*m + i] = V_rowM[i*n + j]; 

        }
    }



    //Operands needed to compute Hn+1
    double *numerator, *denominator_l, *denominator;    //r x n, r x r, r x n


    numerator = malloc(double_size * rn);
    denominator_l = malloc(double_size * rr);

    denominator = malloc(double_size * rn);
    //Operands needed to compute Wn+1
    double *numerator_W, *denominator_W, *denominator_l_W;      // m x r, m x r, m x n
    numerator_W = malloc(double_size * mr);
    denominator_W = malloc(double_size * mr);
    denominator_l_W = malloc(double_size * mn);

    double* approximation; //m x n
    approximation = malloc(double_size * mn);

    double norm_V = 0;
    for (int i = 0; i < mn; i++)
        norm_V += V_rowM[i] * V_rowM[i];
    norm_V = 1 / sqrt(norm_V);

    //real convergence computation
    double err = -1;											
    for (int count = 0; count < maxIteration; count++) {
     
        err = error(approximation, V_rowM, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }    
        
        transpose(W, Wt, m, r);
        //print_matrix_helper(W, r,m);
        int nij;
        int dij;
        double wt;
        
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < r; i++) {
                nij = i * n + j;
                dij = i * r + j;
                numerator[nij] = 0;
                if(j<r){
                    denominator_l[dij] = 0;
                }
                for (int k = 0; k < m; k++){  
                    wt = Wt[i * m + k];
                    numerator[nij] += wt * V_colM[j * m + k];                
                    if(j<r){
                        denominator_l[dij] += wt * Wt[j * m + k];
                    }
                    
                }
            }
        }
        transpose(Wt, W, r, m);
        // print_matrix_helper(V_rowM, m, n);
        // print_matrix_helper(W, m, r );
        // print_matrix_helper(numerator, r, n);
        // print_matrix_helper(denominator_l, r,r );

        //matrix_mul_aopt1(denominator_l, r, r, H, r, n, denominator, r, n);
        
        transpose(H, Ht, r, n);
        matrix_rtrans_mul_aopt1(denominator_l, r, r, Ht, n, r, denominator, r, n);

        for (int i = 0; i < rn; i++)
            H[i] = H[i] * numerator[i] / denominator[i];

        //computation for Wn+1
        matrix_rtrans_mul_aopt1(V_rowM, m, n, H, r, n, numerator_W, m, r);
        matrix_mul_aopt1(W, m, r, H, r, n, denominator_l_W, m, n);
        matrix_rtrans_mul_aopt1(denominator_l_W, m, n, H, r, n, denominator_W, m, r);

        for (int i = 0; i < mr; i++)
            W[i] = W[i] * numerator_W[i] / denominator_W[i];

        // printf("MAtrix at iteration %d\n", count);
        // print_matrix_helper(V_rowM, m, n);
        // print_matrix_helper(W, m, r);
        // print_matrix_helper(H, r, n);
        // print_matrix_helper(numerator_W, m, r);
        // print_matrix_helper(denominator_l_W, m, n);
        // print_matrix_helper(denominator_W, m, r);

    }

    free(numerator);
    free(denominator);
    free(denominator_l);
    free(numerator_W);
    free(denominator_W);
    free(denominator_l_W);
    free(Wt);
    free(V_colM);
    free(approximation);
    return err;
}

