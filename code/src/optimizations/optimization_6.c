#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <optimizations/optimizations_5.h>

#include "cblas.h"


//NEW - optimization done on algopt_1, unrolling from opt_4

typedef unsigned long long myInt64;

static unsigned int double_size = sizeof(double);

static void transpose(double *src, double *dst,  const int N, const int M) {

    //NEW - introduced blocking and simlified index calcs (code motion, strength reduction)
    int nB = BLOCK_SIZE_TRANS;
    int src_i = 0, src_ii;

    //NEW - introduced double loop to avoid calculating DIV and MOD M*N times
    for(int i = 0; i < N; i += nB)
    {
        for(int j = 0; j < M; j += nB)
        {
            src_ii = src_i;
            for(int ii = i; ii < i + nB; ii++)
            {
                for(int jj = j; jj < j + nB; jj++)
                    dst[N*jj + ii] = src[src_ii + jj];
                src_ii += M;
            }
        }
        src_i += nB * M;
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
void matrix_mul_opt6(double *A, int A_n_row, int A_n_col, double*B, int B_n_row, int B_n_col, double*R, int R_n_row, int R_n_col) {

    //NOTE - we need a row of A, whole block of B and 1 element of R in the cache (normalized for the cache line)
    //NOTE - when taking LRU into account, that is 2 rows of A, the whole block of B and 1 row + 1 element of R
    
    memset(R, 0, double_size * R_n_row * R_n_col);

    // cost: B_col*A_col + 2*A_row*A_col*B_col
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A_n_row, B_n_col, A_n_col, 1,
                A, A_n_col, B, B_n_col,
                0, R, R_n_col);
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
void matrix_rtrans_mul_opt6(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col) {
    
    memset(R, 0, double_size * R_n_row * R_n_col);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                A_n_row, B_n_row, A_n_col, 1,
                A, A_n_col, B, B_n_col,
                0, R, B_n_row);
 
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

    matrix_mul_opt6(W, m, r, H, r, n, approx, m, n);

    double norm_approx, temp;
    double temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
    double norm_approx1 = 0;
    double norm_approx2 = 0;
    double norm_approx3 = 0;
    double norm_approx4 = 0; 
    double norm_approx5 = 0; 
    double norm_approx6 = 0;
    double norm_approx7 = 0;
    double norm_approx8 = 0;

    norm_approx = 0;

    int idx_unroll = mn/8;
    int i;
    for (i=0; i<idx_unroll; i+=8){
        temp1 = V[i] - approx[i];
        temp2 = V[i+1] - approx[i+1];
        temp3 = V[i+2] - approx[i+2];
        temp4 = V[i+3] - approx[i+3];
        temp5 = V[i+4] - approx[i+4];
        temp6 = V[i+5] - approx[i+5];
        temp7 = V[i+6] - approx[i+6];
        temp8 = V[i+7] - approx[i+7];

        norm_approx1 += temp1 * temp1;
        norm_approx2 += temp2 * temp2;
        norm_approx3 += temp3 * temp3;
        norm_approx4 += temp4 * temp4;
        norm_approx5 += temp5 * temp5;
        norm_approx6 += temp6 * temp6;
        norm_approx7 += temp7 * temp7;
        norm_approx8 += temp8 * temp8;

    }

    norm_approx = norm_approx1 + norm_approx2 + norm_approx3 + norm_approx4 + norm_approx5 + norm_approx6 + norm_approx7 + norm_approx8;

    for (; i < mn; i++)
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
double nnm_factorization_opt6(double *V_rowM, double*W, double*H, int m, int n, int r, int maxIteration, double epsilon) {
    double *Htmp;
    double *Wtmp;
    double *V_colM;
    double *Wt, *Ht;
    double *tmp;
    int rn, rr, mr, mn;
    rn = r * n;
    rr = r * r;
    mr = m * r;
    mn = m * n;
    Wtmp   = malloc(double_size * mr);
    Htmp   = malloc(double_size * rn);
    Wt   = malloc(double_size * mr);
    Ht   = malloc(double_size * rn);
    V_colM = malloc(double_size * mn);

    int nB = BLOCK_SIZE_MMUL;

    // this is required to be done here to reuse the same run_opt.
    // does not change the number of flops
    for (int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
           V_colM[j*m + i] = V_rowM[i*n + j]; 

        }
    }


    //Operands needed to compute Hn+1
    double *numerator;
    double *denominator_l;
    double *denominator;    //r x n, r x r, r x n

    numerator       = malloc(double_size * rn);
    denominator_l   = malloc(double_size * rr);
    denominator     = malloc(double_size * rn);

    //Operands needed to compute Wn+1
    double *numerator_W;
    double *denominator_W;
    double *denominator_l_W;      // m x r, m x r, m x n


    numerator_W     = malloc(double_size * mr);
    denominator_W   = malloc(double_size * mr);
    denominator_l_W = malloc(double_size * mn);

    double* approximation; //m x n
    approximation = malloc(double_size * mn);
    

    double norm_V  = 0;
    double norm_V1 = 0;
    double norm_V2 = 0;
    double norm_V3 = 0;
    double norm_V4 = 0; 
    double norm_V5 = 0; 
    double norm_V6 = 0;
    double norm_V7 = 0;
    double norm_V8 = 0;


    int idx_unroll = mn/8;
    int i;

    ///// NORM

    for (i=0; i<idx_unroll; i+=8){
        norm_V1 += V_rowM[i]   * V_rowM[i];
        norm_V2 += V_rowM[i+1] * V_rowM[i+1];
        norm_V3 += V_rowM[i+2] * V_rowM[i+2];
        norm_V4 += V_rowM[i+3] * V_rowM[i+3];
        norm_V5 += V_rowM[i+4] * V_rowM[i+4];
        norm_V6 += V_rowM[i+5] * V_rowM[i+5];
        norm_V7 += V_rowM[i+6] * V_rowM[i+6];
        norm_V8 += V_rowM[i+7] * V_rowM[i+7];
    }

    norm_V = norm_V1 + norm_V2 + norm_V3 + norm_V4 + norm_V5 + norm_V6 + norm_V7 + norm_V8;

    for (; i < mn; i++)
    {
        norm_V += V_rowM[i] * V_rowM[i];
    }

    norm_V = 1 / sqrt(norm_V);

    ///// END NORM


    double err = -1;											
    for (int count = 0; count < maxIteration; count++) {
     
        err = error(approximation, V_rowM, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }    
        
        transpose(W, Wt, m, r);
        //print_matrix_helper(W, r,m);

        memset(denominator_l, 0, double_size * r * r);
        memset(numerator, 0, double_size * r * n);
        //NEW blocking using blas for inner multiplication
        for (int j = 0; j < n; j+= nB) {
            
            for (int i = 0; i < r; i+= nB) {
                
                for (int k = 0; k < m; k+= nB){  
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            nB, nB, nB, 1,
                            (&(Wt[i*m + k])), m, &(V_rowM[k*n + j]), n,
                            1, &(numerator[i*n + j]), n);


                    if(j<r){
                          
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            nB, nB, nB, 1,
                            (&(Wt[i*m + k])), m, &(W[k*r + j]), r,
                            1, &(denominator_l[i*r + j]), r);                    
                    }
                    
                }
            }
        }
        transpose(Wt, W, r, m);
        transpose(H, Ht, r, n);
        int Rij, Rii;

        memset(denominator, 0, double_size * r * n);
       
        for (int i = 0; i < r; i+= nB) {
            for (int j = 0; j < n; j+= nB) {
                for (int k = 0; k < r; k+= nB){
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        nB, nB, nB, 1,
                        (&(denominator_l[i*r + k])), r,
                         &(H[k*n + j]), n,
                        1, &(denominator[i*n + j]), n); 
                }
                Rii = i * n + j;
                for (int ii = 0; ii < nB; ii++) {
                    for (int jj = 0; jj < nB; jj++) {
                        Rij = Rii + jj;
                        Htmp[Rij] = H[Rij] * numerator[Rij] / denominator[Rij];
                    }
                    Rii += n;
                }
            }
        }
  


        
        tmp = Htmp;
        Htmp = H;
        H = tmp;  

        transpose(H, Ht, r, n);

        memset(denominator_l, 0, double_size * r * r);
        memset(numerator_W, 0, double_size * m * r);

        //NEW blocking using blas for inner multiplication
        for (int i = 0; i < m; i += nB) {
            for (int j = 0; j < r; j+= nB) {
                for (int k = 0; k < n; k+= nB){

                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            nB, nB, nB, 1,
                            &(V_rowM[i*n + k]), n,
                            &(Ht[k*r + j]), r,
                            1, &(numerator_W[i*r + j]), r);

                    if(i < r){

                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            nB, nB, nB, 1,
                            &(H[i*n + k]), n,
                            &(Ht[k*r + j]), r,
                            1, &(denominator_l[i*r + j]), r);

                    }
                }
            }
        }

        memset(denominator_W, 0, double_size * m * r);
        for (int i = 0; i < m; i+= nB) {
            for (int j = 0; j < r; j+= nB) {
                for (int k = 0; k < r; k+= nB){
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        nB, nB, nB, 1,
                        (&(W[i*r + k])), r,
                         &(denominator_l[k*r + j]), r,
                        1, &(denominator_W[i*r + j]), r); 
                }
                Rii = i*r + j;
                for (int ii = 0; ii < nB; ii++) {
                    for (int jj = 0; jj < nB; jj++) {
                        Rij = Rii + jj;
                        Wtmp[Rij] = W[Rij] * numerator_W[Rij] / denominator_W[Rij];
                    }
                    Rii += r;
                }
       
            }
     
        }

        tmp = Wtmp;
        Wtmp = W;
        W = tmp; 

    }
    free(Wtmp);
    free(Htmp);
    free(numerator);
    free(denominator);
    free(denominator_l);
    free(numerator_W);
    free(denominator_W);
    free(denominator_l_W);
    free(Wt);
    free(Ht);
    free(V_colM);
    free(approximation);
    return err;
}

