#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <optimizations/optimizations_37.h>

//NEW - optimization done on optimization_33 - Introducing rectangular blocks

typedef unsigned long long myInt64;

static unsigned int double_size = sizeof(double);

static void transpose(double *src, double *dst,  const int N, const int M) {

    int nB = BLOCK_SIZE_TRANS;
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
void matrix_mul_opt37(double *A, int A_n_row, int A_n_col, double*B, int B_n_row, int B_n_col, double*R, int R_n_row, int R_n_col) {

    //NOTE - we need a row of A, whole block of B and 1 element of R in the cache (normalized for the cache line)
    //NOTE - when taking LRU into account, that is 2 rows of A, the whole block of B and 1 row + 1 element of R
    
    int Rij = 0, Ri = 0, Ai = 0, Aii, Rii;
    int nB = BLOCK_SIZE_MMUL;
    int nBR_n_col = nB * R_n_col;
    int nBA_n_col = nB * A_n_col;

    double R_Rij;

    memset(R, 0, double_size * R_n_row * R_n_col);

    for (int i = 0; i < A_n_row; i+=nB) {
        for (int j = 0; j < B_n_col; j+=nB) {
            for (int k = 0; k < A_n_col; k+=nB) {
                Rii = Ri;
                Aii = Ai;
                for (int ii = i; ii < i + nB; ii++) {
                    for (int jj = j; jj < j + nB; jj++) {
                        Rij = Rii + jj;
                        R_Rij = 0;
                        for (int kk = k; kk < k + nB; kk++)
                            R_Rij += A[Aii + kk] * B[kk * B_n_col + jj];
                        R[Rij] += R_Rij;
                    }
                    Rii += R_n_col;
                    Aii += A_n_col;
                }
            }
        }
        Ri += nBR_n_col;
        Ai += nBA_n_col;
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
void matrix_rtrans_mul_opt37(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col) {
    
    int Rij = 0, Ri = 0, Ai = 0, Bj, Rii, Aii, Bjj;
    int nB = BLOCK_SIZE_RTRANSMUL;
    int nBR_n_col = nB * R_n_col;
    int nBA_n_col = nB * A_n_col;
    int nBB_n_col = nB * B_n_col;

    double R_Rij;

    memset(R, 0, double_size * R_n_row * R_n_col);

    for (int i = 0; i < A_n_row; i+=nB) {
        Bj = 0;
        for (int j = 0; j < B_n_row; j+=nB) {
            for (int k = 0; k < A_n_col; k+=nB){
                Aii = Ai;
                Rii = Ri;
                for (int ii = i; ii < i + nB; ii++) {
                    Bjj = Bj;
                    for (int jj = j; jj < j + nB; jj++) {
                        Rij = Rii + jj;
                        R_Rij = 0;
                        for (int kk = k; kk < k + nB; kk++)
                            R_Rij += A[Aii + kk] * B[Bjj + kk];
                        R[Rij] += R_Rij;
                        Bjj += B_n_col;
                    }
                    Aii += A_n_col;
                    Rii += R_n_col;
                }
            }
            Bj += nBB_n_col;
        }
        Ai += nBA_n_col;
        Ri += nBR_n_col;
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

    matrix_mul_opt37(W, m, r, H, r, n, approx, m, n);

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
double nnm_factorization_opt37(double *V, double*W, double*H, int m, int n, int r, int maxIteration, double epsilon) {
    double *Wt, *H_new;
    int rn, rr, mr, mn;
    rn = r * n;
    rr = r * r;
    mr = m * r;
    mn = m * n;

    int d_rn, d_rr, d_mr, d_mn;
    d_rn = double_size * rn;
    d_rr = double_size * rr;
    d_mr = double_size * mr;
    d_mn = double_size * mn;
    
    Wt = malloc(d_mr);
    H_new = malloc(d_rn);

    //Operands needed to compute Hn+1
    double *numerator;      //r x n
    double *denominator_l;  //r x r
    double *denominator;    //r x n

    numerator = malloc(d_rn);
    denominator_l = malloc(d_rr);
    denominator = malloc(d_rn);

    //Operands needed to compute Wn+1
    double *numerator_W;    //m x r
    double* denominator_r;  //r x r
    double *denominator_W;  //m x r

    numerator_W = malloc(d_mr);
    denominator_r = malloc(d_rr);
    denominator_W = malloc(d_mr);

    double* approximation; //m x n
    approximation = malloc(d_mn);

    double norm_V = 0;
    for (int i = 0; i < mn; i++){
        norm_V += V[i] * V[i];
    }
    norm_V = 1 / sqrt(norm_V);

    //NOTE - We may also try calling BLAS on the level of blocks

    int nB_i = BLOCK_SIZE_H_ROW;
    int nB_j = BLOCK_SIZE_H_COL;
    int nB_mul = BLOCK_SIZE_MMUL;
    int inB, jnB, mnB_mul = m * nB_mul, mnB_i = m * nB_i, rnB_i = r * nB_i, nnB_i = n * nB_i;
    int ri, mi, ni, ri1, mi1, ni1, nj1, ni1j1, ri1j1, ri1jj1, mj1, mjj1;

    double accumulator;

    //real convergence computation
    double err = -1;											
    for (int count = 0; count < maxIteration; count++) {
     
        err = error(approximation, V, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }    
        
        memset(denominator_l, 0, d_rr);
        memset(numerator, 0, d_rn);
        memset(denominator, 0, d_rn);

        memset(numerator_W, 0, d_mr);
        memset(denominator_r, 0, d_rr);

        transpose(W, Wt, m, r);

        ri = mi = ni = 0;
        for (int i = 0; i < r; i += nB_i) {
            inB = i + nB_i;

            //computation for Hn+1
            
            //We need the whole row of WtW for calculating a single block of H, so we calculate the row only once and use it for all blocks of H with the same i
            //Wt*Wt rmul
            ri1 = ri, mi1 = mi;
            for (int i1 = i; i1 < inB; i1++) {
                mj1 = 0;
                for (int j1 = 0; j1 < r; j1 += nB_mul) {
                    for (int k1 = 0; k1 < m; k1 += nB_mul) {
                        mjj1 = mj1;
                        for (int jj1 = j1; jj1 < j1 + nB_mul; jj1++) {
                            ri1jj1 = ri1 + jj1;
                            accumulator = 0;
                            for (int kk1 = k1; kk1 < k1 + nB_mul; kk1++)
                                accumulator += Wt[mi1 + kk1] * Wt[mjj1 + kk1];
                            denominator_l[ri1jj1] += accumulator;
                            mjj1 += m;
                        }
                    }
                    mj1 += mnB_mul;
                }
                ri1 += r;
                mi1 += m;
            }

            for (int j = 0; j < n; j += nB_j) {
                jnB = j + nB_j;
                
                //computation for Hn+1
                
                //Wt*V mul
                mi1 = mi;
                ni1 = ni;
                for (int i1 = i; i1 < inB; i1++) {
                    for (int j1 = j; j1 < jnB; j1++) {
                        ni1j1 = ni1 + j1;
                        accumulator = 0;
                        for (int k1 = 0; k1 < m; k1++)
                            accumulator += Wt[mi1 + k1] * V[k1 * n + j1];
                        numerator[ni1j1] += accumulator;
                    }
                    mi1 += m;
                    ni1 += n;
                }

                //(WtW)*H mul
                ni1 = ni;
                ri1 = ri;
                for (int i1 = i; i1 < inB; i1++) {
                    for (int j1 = j; j1 < jnB; j1++) {
                        ni1j1 = ni1 + j1;
                        accumulator = 0;
                        for (int k1 = 0; k1 < r; k1++)
                            accumulator += denominator_l[ri1 + k1] * H[k1 * n + j1];
                        denominator[ni1j1] += accumulator;
                    }
                    ni1 += n;
                    ri1 += r;
                }
                
                //element-wise multiplication and division
                ni1 = ni;
                for (int i1 = i; i1 < inB; i1++) {
                    for (int j1 = j; j1 < jnB; j1++) {
                        ni1j1 = ni1 + j1;
                        H_new[ni1j1] = H[ni1j1] * numerator[ni1j1] / denominator[ni1j1];
                    }
                    ni1 += n;
                }


                //computation for Wn+1

                //V*H rmul
                ri1 = ni1 = 0;
                for (int i1 = 0; i1 < m; i1++) {
                    nj1 = ni;
                    for (int j1 = i; j1 < inB; j1++) {
                        ri1j1 = ri1 + j1;
                        accumulator = 0;
                        for (int k1 = j; k1 < jnB; k1++)
                            accumulator += V[ni1 + k1] * H_new[nj1 + k1];
                        numerator_W[ri1j1] += accumulator;
                        nj1 += n;
                    }
                    ri1 += r;
                    ni1 += n;
                }

                //H*H rmul
                ni1 = ri1 = 0;
                for (int i1 = 0; i1 < inB; i1++) {
                    nj1 = ni;
                    for (int j1 = i; j1 < inB; j1++) {
                        ri1j1 = ri1 + j1;
                        accumulator = 0;
                        for (int k1 = j; k1 < jnB; k1++)
                            accumulator += H_new[ni1 + k1] * H_new[nj1 + k1];
                        denominator_r[ri1j1] += accumulator;
                        nj1 += n;
                    }
                    ni1 += n;
                    ri1 += r;
                }
                ni1 = ni;
                ri1 = ri;
                for (int i1 = i; i1 < inB; i1++) {
                    nj1 = 0;
                    for (int j1 = 0; j1 < i; j1++) {
                        ri1j1 = ri1 + j1;
                        accumulator = 0;
                        for (int k1 = j; k1 < jnB; k1++)
                            accumulator += H_new[ni1 + k1] * H_new[nj1 + k1];
                        denominator_r[ri1j1] += accumulator;
                        nj1 += n;
                    }
                    ri1 += r;
                    ni1 += n;
                }
            }
            ri += rnB_i;
            mi += mnB_i;
            ni += nnB_i;
        }

        //remaining computation for Wn+1
        matrix_rtrans_mul_opt37(W, m, r, denominator_r, r, r, denominator_W, m, r);

        for (int i = 0; i < mr; i++)
            W[i] = W[i] * numerator_W[i] / denominator_W[i];

        memcpy(H, H_new, d_rn);
    }

    free(numerator);
    free(denominator);
    free(denominator_l);
    free(denominator_r);
    free(numerator_W);
    free(denominator_W);
    free(Wt);
    free(H_new);
    free(approximation);
    return err;
}

