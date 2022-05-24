#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <optimizations/optimizations_2.h>

#include "cblas.h"

//NEW - optimization of the algorithm using the function optimizations for optimization 3, nnmf based on alg opt 1

typedef unsigned long long myInt64;

static unsigned int double_size = sizeof(double);

static void transpose(double *src, double *dst, const int N, const int M) {

    int nB = BLOCK_SIZE_TRANS;
    int src_i = 0, src_ii;

    for (int i = 0; i < N; i += nB) {
        for (int j = 0; j < M; j += nB) {
            src_ii = src_i;
            for (int ii = i; ii < i + nB; ii++) {
                for (int jj = j; jj < j + nB; jj++)
                    dst[N * jj + ii] = src[src_ii + jj];
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
void matrix_mul_opt11(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R, int R_n_row,
                      int R_n_col) {

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A_n_row, B_n_col, A_n_col, 1,
                A, A_n_col, B, B_n_col,
                0, R, B_n_col);
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
void matrix_rtrans_mul_opt11(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R,
                             int R_n_row, int R_n_col) {

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                A_n_row, B_n_row, A_n_col, 1,
                A, A_n_col, B, B_n_col,
                0, R, B_n_row);
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
void matrix_ltrans_mul_opt11(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R,
                             int R_n_row, int R_n_col) {

    //NEW using BLAS
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                A_n_col, B_n_col, B_n_row, 1, //r=A->n_row = B->n_row
                A, A_n_col, B, B_n_col,
                0, R, B_n_col);
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
inline double error(double *approx, double *V, double *W, double *H, int m, int n, int r, int mn, double norm_V) {

    matrix_mul_opt11(W, m, r, H, r, n, approx, m, n);

    double norm_approx, temp;

    norm_approx = 0;
    for (int i = 0; i < mn; i++) {
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
double
nnm_factorization_opt11(double *V, double *W, double *H, int m, int n, int r, int maxIteration, double epsilon) {

    int nB = BLOCK_SIZE_NNMF;
    double *Wtmp, *Ht, *tmp, *denominator_r, *W_new, *numerator_H, *denominator_H;
    double *V_colM;
    int rn, rr, mr, mn;
    rn = r * n;
    rr = r * r;
    mr = m * r;
    mn = m * n;

    Wtmp = malloc(double_size * mr);
    W_new = malloc(double_size * mr);
    numerator_H = malloc(double_size * rr);
    denominator_H = malloc(double_size * rn);
    Ht = malloc(double_size * rn);
    V_colM = malloc(double_size * mn);
    denominator_r = malloc(double_size * mn);

    int d_rn, d_rr;
    d_rn = double_size * rn;
    d_rr = double_size * rr;

    double *numerator, *denominator_l, *denominator;    //r x n, r x r, r x n

    numerator = malloc(double_size * rn);
    denominator_l = malloc(double_size * rr);

    denominator = malloc(double_size * rn);

    double *approximation; //m x n
    approximation = malloc(double_size * mn);

    int idx_unroll = 0;

    double norm_V = 0;
    for (int i = 0; i < mn; i++)
        norm_V += V[i] * V[i];
    norm_V = 1 / sqrt(norm_V);

    double err = -1;

    //NEW: Calculating the first iteration of H outside of the iterable as it is not interleaved, using bs2
    //NOTE: Based on opt24
    //computation for H0
    matrix_ltrans_mul_opt11(W, m, r, V, m, n, numerator, r, n);
    matrix_ltrans_mul_opt11(W, m, r, W, m, r, denominator_l, r, r);
    matrix_mul_opt11(denominator_l, r, r, H, r, n, denominator, r, n);

    idx_unroll = rn / 8;
    int i;
    for (i = 0; i < idx_unroll; i += 8) {
        H[i] = H[i] * numerator[i] / denominator[i];
        H[i + 1] = H[i + 1] * numerator[i + 1] / denominator[i + 1];
        H[i + 2] = H[i + 2] * numerator[i + 2] / denominator[i + 2];
        H[i + 3] = H[i + 3] * numerator[i + 3] / denominator[i + 3];
        H[i + 4] = H[i + 4] * numerator[i + 4] / denominator[i + 4];
        H[i + 5] = H[i + 5] * numerator[i + 5] / denominator[i + 5];
        H[i + 6] = H[i + 6] * numerator[i + 6] / denominator[i + 6];
        H[i + 7] = H[i + 7] * numerator[i + 7] / denominator[i + 7];
    }
    for (; i < rn; i++) {
        H[i] = H[i] * numerator[i] / denominator[i];
    }

    int inB, jnB, mnB = m * nB, rnB = r * nB, nnB = n * nB;
    int ri, mi, ni, ri1, mi1, ni1, nj1, ni1j1, ri1j1, ri1jj1, mj1, mjj1;

    for (int count = 0; count < maxIteration; count++) {

        memset(denominator_l, 0, d_rr);
        memset(numerator, 0, d_rn);
        memset(denominator, 0, d_rn);

        memset(numerator_H, 0, d_rn);
        memset(denominator_r, 0, d_rr);

        transpose(H, Ht, m, r);
        ri = mi = ni = 0;
        for (i = 0; i < r; i += nB) {
            inB = i + nB;
            for (int j = 0; j < n; j += nB) {
                jnB = j + nB;

                //computation for Wn+1

                //Ht*Ht rmul
                if (j == 0)
                {
                    ri1 = ri, mi1 = mi;
                    for (int i1 = i; i1 < inB; i1++) {
                        mj1 = 0;
                        for (int j1 = 0; j1 < n; j1 += nB) {
                            for (int k1 = 0; k1 < r; k1 += nB) {
                                mjj1 = mj1;
                                for (int jj1 = j1; jj1 < j1 + nB; jj1++) {
                                    ri1jj1 = ri1 + jj1;
                                    for (int kk1 = k1; kk1 < k1 + nB; kk1++)
                                        denominator_r[ri1jj1] += Ht[mi1 + kk1] * Ht[mjj1 + kk1];
                                    mjj1 += r;
                                }
                            }
                            mj1 += mnB;
                        }
                        ri1 += n;
                        mi1 += r;
                    }
                }

                //V*Ht mul
                mi1 = mi;
                ni1 = ni;
                for (int i1 = i; i1 < inB; i1++) {
                    for (int j1 = j; j1 < jnB; j1++) {
                        ni1j1 = ni1 + j1;
                        for (int k1 = 0; k1 < m; k1++)
                            numerator[ni1j1] += V[mi1 + k1] * Ht[k1 * n + j1];
                    }
                    mi1 += m;
                    ni1 += n;
                }

                //W*(HHt) mul
                ni1 = ni;
                ri1 = ri;
                for (int i1 = i; i1 < inB; i1++) {
                    for (int j1 = j; j1 < jnB; j1++) {
                        ni1j1 = ni1 + j1;
                        for (int k1 = 0; k1 < r; k1++)
                            denominator[ni1j1] +=  W[k1 * m + j1] * denominator_r[ri1 + k1];
                    }
                    ni1 += n;
                    ri1 += r;
                }

                //element-wise multiplication and division
                ni1 = ni;
                for (int i1 = i; i1 < inB; i1++) {
                    for (int j1 = j; j1 < jnB; j1++) {
                        ni1j1 = ni1 + j1;
                        W_new[ni1j1] = W[ni1j1] * numerator[ni1j1] / denominator[ni1j1];
                    }
                    ni1 += n;
                }

                //Wt*V rmul
                ri1 = ni1 = 0;
                for (int i1 = 0; i1 < n; i1++) {
                    nj1 = ni;
                    for (int j1 = i; j1 < inB; j1++) {
                        ri1j1 = ri1 + j1;
                        for (int k1 = j; k1 < jnB; k1++)
                            numerator_H[ri1j1] += W_new[nj1 + k1] * V[ni1 + k1];
                        nj1 += m;
                    }
                    ri1 += r;
                    ni1 += m;
                }

                //W*W rmul
                ni1 = ri1 = 0;
                for (int i1 = 0; i1 < inB; i1++) {
                    nj1 = ni;
                    for (int j1 = i; j1 < inB; j1++) {
                        ri1j1 = ri1 + j1;
                        for (int k1 = j; k1 < jnB; k1++) {
                            denominator_l[ri1j1] += W_new[ni1 + k1] * W_new[nj1 + k1];
                        }
                        nj1 += m;
                    }
                    ni1 += m;
                    ri1 += r;
                }
                ni1 = ni;
                ri1 = ri;
                for (int i1 = i; i1 < inB; i1++) {
                    nj1 = 0;
                    for (int j1 = 0; j1 < i; j1++) {
                        ri1j1 = ri1 + j1;
                        for (int k1 = j; k1 < jnB; k1++) {
                            denominator_l[ri1j1] += W_new[ni1 + k1] * W_new[nj1 + k1];
                        }
                        nj1 += m;
                    }
                    ri1 += r;
                    ni1 += m;
                }
                ri += rnB;
                mi += mnB;
                ni += nnB;
            }
        }

        printf("TEST\n");

        tmp = W_new;
        W_new = W;
        W = tmp;

        //NOTE: Need to check at the end as the first iteration is interleaved with the second already
        err = error(approximation, V, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }

        matrix_ltrans_mul_opt11(W, m, r, denominator_r, m, n, denominator_H, r, n);

        idx_unroll = rn / 8;
        for (i = 0; i < idx_unroll; i += 8) {
            H[i] = H[i] * numerator_H[i] / denominator[i];
            H[i + 1] = H[i + 1] * numerator_H[i + 1] / denominator_H[i + 1];
            H[i + 2] = H[i + 2] * numerator_H[i + 2] / denominator_H[i + 2];
            H[i + 3] = H[i + 3] * numerator_H[i + 3] / denominator_H[i + 3];
            H[i + 4] = H[i + 4] * numerator_H[i + 4] / denominator_H[i + 4];
            H[i + 5] = H[i + 5] * numerator_H[i + 5] / denominator_H[i + 5];
            H[i + 6] = H[i + 6] * numerator_H[i + 6] / denominator_H[i + 6];
            H[i + 7] = H[i + 7] * numerator_H[i + 7] / denominator_H[i + 7];
        }
        for (; i < rn; i++) {
            H[i] = H[i] * numerator_H[i] / denominator_H[i];
        }
    }

    free(numerator);
    free(denominator);
    free(denominator_l);
    free(Wtmp);
    free(V_colM);
    free(approximation);
    free(Ht);
    return err;
}


