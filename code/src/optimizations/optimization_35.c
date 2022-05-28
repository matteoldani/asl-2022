#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <optimizations/optimizations_35.h>

//NEW - optimization done on optimization_3 - Introduced algorithic changes to nnmf - calculate W block by block and reuse instantly

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
void matrix_mul_opt35(double *A, int A_n_row, int A_n_col, double*B, int B_n_row, int B_n_col, double*R, int R_n_row, int R_n_col) {

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
void matrix_rtrans_mul_opt35(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col) {
    
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

    matrix_mul_opt35(W, m, r, H, r, n, approx, m, n);

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
double nnm_factorization_opt35(double *V, double*W, double*H, int m, int n, int r, int maxIteration, double epsilon) {
    double *Wt, *H_new, *W_new;
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
    W_new = malloc(d_mr);

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

    //NEW - Algorithmic optimization, calculating W(n+1) in a blockwise manner and using the current block instantly in the calculation of H(n+2)
    //NEW - All multiplications are done in the most optimal manner - blocked and with index calculation optimizations and scalar replacement
    //NOTE - We may also try calling BLAS on the level of blocks

    int nB = BLOCK_SIZE_W;
    int inB, jnB, mnB = m * nB, rnB = r * nB, nnB = n * nB;
    int ri, mi, ni, nj, ri1, mi1, ni1, nj1, ni1j1, ri1j1, ri1jj1, mj1, mjj1;

    double accumulator;

    //NEW - precompute first H so we can start with calculating W blockwise and reusing blocks for next H
    //computation for H1
    transpose(W, Wt, m, r);
    matrix_mul_opt35(Wt, r, m, V, m, n, numerator, r, n);
    matrix_rtrans_mul_opt35(Wt, r, m, Wt, r, m, denominator_l, r, r);
    matrix_mul_opt35(denominator_l, r, r, H, r, n, denominator, r, n);

    for (int i = 0; i < rn; i++)
        H_new[i] = H[i] * numerator[i] / denominator[i];

    //real convergence computation
    double err = -1;											
    for (int count = 0; count < maxIteration; count++) {
     
        err = error(approximation, V, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }    
        
        /*memset(denominator_l, 0, d_rr);
        memset(numerator, 0, d_rn);
        memset(denominator, 0, d_rn);

        memset(numerator_W, 0, d_mr);
        memset(denominator_r, 0, d_rr);

        transpose(W, Wt, m, r);

        ri = mi = ni = 0;
        for (int i = 0; i < r; i += nB) {
            inB = i + nB;
            for (int j = 0; j < n; j += nB) {
                jnB = j + nB;
                
                //computation for Hn+1

                //Wt*Wt rmul
                if (j == 0)
                {
                    ri1 = ri, mi1 = mi;
                    for (int i1 = i; i1 < inB; i1++) {
                        mj1 = 0;
                        for (int j1 = 0; j1 < r; j1 += nB) {
                            for (int k1 = 0; k1 < m; k1 += nB) {
                                mjj1 = mj1;
                                for (int jj1 = j1; jj1 < j1 + nB; jj1++) {
                                    ri1jj1 = ri1 + jj1;
                                    accumulator = 0;
                                    for (int kk1 = k1; kk1 < k1 + nB; kk1++)
                                        accumulator += Wt[mi1 + kk1] * Wt[mjj1 + kk1];
                                    denominator_l[ri1jj1] += accumulator;
                                    mjj1 += m;
                                }
                            }
                            mj1 += mnB;
                        }
                        ri1 += r;
                        mi1 += m;
                    }
                }

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

                //computation for Wn+1

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
            ri += rnB;
            mi += mnB;
            ni += nnB;
        }

        matrix_rtrans_mul_opt35(W, m, r, denominator_r, r, r, denominator_W, m, r);

        for (int i = 0; i < mr; i++)
            W[i] = W[i] * numerator_W[i] / denominator_W[i];

        memcpy(H, H_new, d_rn);*/
        
        memcpy(H, H_new, d_rn);
        memset(numerator_W, 0, d_mr);
        memset(denominator_l, 0, d_rr);
        memset(denominator_W, 0, d_mr);

        matrix_rtrans_mul_opt35(H, r, n, H, r, n, denominator_l, r, r);
        matrix_mul_opt35(W, m, r, denominator_l, r, r, denominator_W, m, r);

        ri = mi = ni = 0;
        for (int i = 0; i < m; i += nB) {
            inB = i + nB;
            nj = 0;
            for (int j = 0; j < r; j += nB) {
                jnB = j + nB;

                //computation for Wn+1

                /*//Wt*Wt rmul
                if (j == 0)
                {
                    ri1 = ri, mi1 = mi;
                    for (int i1 = i; i1 < inB; i1++) {
                        mj1 = 0;
                        for (int j1 = 0; j1 < r; j1 += nB) {
                            for (int k1 = 0; k1 < m; k1 += nB) {
                                mjj1 = mj1;
                                for (int jj1 = j1; jj1 < j1 + nB; jj1++) {
                                    ri1jj1 = ri1 + jj1;
                                    accumulator = 0;
                                    for (int kk1 = k1; kk1 < k1 + nB; kk1++)
                                        accumulator += Wt[mi1 + kk1] * Wt[mjj1 + kk1];
                                    denominator_l[ri1jj1] += accumulator;
                                    mjj1 += m;
                                }
                            }
                            mj1 += mnB;
                        }
                        ri1 += r;
                        mi1 += m;
                    }
                }*/

                //VH rmul
                ni1 = ni;
                ri1 = ri;
                for (int i1 = i; i1 < inB; i1++) {
                    nj1 = nj;
                    for (int j1 = j; j1 < jnB; j1++) {
                        accumulator = 0;
                        for (int k1 = 0; k1 < n; k1++)
                            accumulator += V[ni1 + k1] * H[nj1 + k1];
                        numerator_W[ri1 + j1] += accumulator;
                        nj1 += n;
                    }
                    ni1 += n;
                    ri1 += r;
                }

                /*//(WtW)*H mul
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
                }*/

                //element-wise multiplication and division
                ri1 = ri;
                for (int i1 = i; i1 < inB; i1++) {
                    for (int j1 = j; j1 < jnB; j1++) {
                        ri1j1 = ri1 + j1;
                        W_new[ri1j1] = W[ri1j1] * numerator_W[ri1j1] / denominator_W[ri1j1];
                    }
                    ri1 += r;
                }

                nj += nnB;
            }
            ri += rnB;
            mi += mnB;
            ni += nnB;
        }

        //computation for Wn+1
        //matrix_rtrans_mul_opt35(V, m, n, H, r, n, numerator_W, m, r);
        //matrix_rtrans_mul_opt35(H, r, n, H, r, n, denominator_l, r, r);
        //matrix_mul_opt35(W, m, r, denominator_l, r, r, denominator_W, m, r);

        /*for (int i = 0; i < mr; i++)
            W[i] = W[i] * numerator_W[i] / denominator_W[i];*/

        //computation for Hn+2
        transpose(W, Wt, m, r);
        matrix_mul_opt35(Wt, r, m, V, m, n, numerator, r, n);
        matrix_rtrans_mul_opt35(Wt, r, m, Wt, r, m, denominator_l, r, r);
        matrix_mul_opt35(denominator_l, r, r, H, r, n, denominator, r, n);

        for (int i = 0; i < rn; i++)
            H_new[i] = H[i] * numerator[i] / denominator[i];
    }

    free(numerator);
    free(denominator);
    free(denominator_l);
    free(denominator_r);
    free(numerator_W);
    free(denominator_W);
    free(Wt);
    free(H_new);
    free(W_new);
    free(approximation);
    return err;
}

