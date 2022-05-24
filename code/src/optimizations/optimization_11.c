#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <optimizations/optimizations_2.h>

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

    //NOTE - we need a row of A, whole block of B and 1 element of R in the cache (normalized for the cache line)
    //NOTE - when taking LRU into account, that is 2 rows of A, the whole block of B and 1 row + 1 element of R

    int Rij = 0, Ri = 0, Ai = 0, Aii, Rii;
    int nB = BLOCK_SIZE_MMUL;

    double R_Rij;

    memset(R, 0, double_size * R_n_row * R_n_col);

    for (int i = 0; i < A_n_row; i += nB) {
        for (int j = 0; j < B_n_col; j += nB) {
            for (int k = 0; k < A_n_col; k += nB) {
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
        Ri += nB * R_n_col;
        Ai += nB * A_n_col;
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
void
matrix_rtrans_mul_opt11(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R,
                        int R_n_row,
                        int R_n_col) {

    int Rij = 0, Ri = 0, Ai = 0, Bj, Rii, Aii, Bjj;
    int nB = BLOCK_SIZE_RTRANSMUL;

    double R_Rij;

    memset(R, 0, double_size * R_n_row * R_n_col);

    for (int i = 0; i < A_n_row; i += nB) {
        Bj = 0;
        for (int j = 0; j < B_n_row; j += nB) {
            for (int k = 0; k < A_n_col; k += nB) {
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
            Bj += nB * B_n_col;
        }
        Ai += nB * A_n_col;
        Ri += nB * R_n_col;
    }
}

/**
 * @brief compute the multiplication of A transposed and B
 * @param A is the matrix to be transposed
 * @param B is the other factor of the multiplication
 * @param R is the matrix that will hold the result
 */

// ColMajor impl  ----Param num 8 has an illegal value (0.327659)
void matrix_rtrans_mul_bs2_cm(Matrix *A, Matrix *B, Matrix *R) {

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                A->n_row, B->n_row, A->n_col, 1,
                A->M, A->n_row, B->M, B->n_col,
                0, R->M, A->n_row);
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
nnm_factorization_opt11(double *V_rowM, double *W, double *H, int m, int n, int r, int maxIteration, double epsilon) {

    double *Wt, *Wtmp, *Ht, *Htmp, *tmp;
    double *V_colM;
    int rn, rr, mr, mn;
    rn = r * n;
    rr = r * r;
    mr = m * r;
    mn = m * n;
    Wt = malloc(double_size * mr);
    Wtmp = malloc(double_size * mr);
    Ht = malloc(double_size * rn);
    Htmp = malloc(double_size * rn);
    V_colM = malloc(double_size * mn);

    int nB = BLOCK_SIZE_NNMF;

    // this is required to be done here to reuse the same run_opt.
    // does not change the number of flops
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            V_colM[j * m + i] = V_rowM[i * n + j];
        }
    }

    double *numerator, *denominator_l, *denominator;    //r x n, r x r, r x n

    numerator = malloc(double_size * rn);
    denominator_l = malloc(double_size * rr);

    denominator = malloc(double_size * rn);

    double *numerator_W, *denominator_W, *denominator_l_W;      // m x r, m x r, m x n
    numerator_W = malloc(double_size * mr);
    denominator_W = malloc(double_size * mr);
    denominator_l_W = malloc(double_size * mn);

    double *approximation; //m x n
    approximation = malloc(double_size * mn);

    double norm_V = 0;
    for (int i = 0; i < mn; i++)
        norm_V += V_rowM[i] * V_rowM[i];
    norm_V = 1 / sqrt(norm_V);

    double err = -1;

    //NEW: Calculating the first iteration of H outside of the iterable as it is not interleaved, using bs2
    // Calculating first iteration of H without interleaving it with W
    //computation for Hn+1
    matrix_ltrans_mul_(W, V, &numerator);          // cost: B_col*B_row + 2*B_row*A_col*B_col = n*m + 2*m*r*n
    matrix_ltrans_mul_bs2(W, W, &denominator_l);      // cost: B_col*B_row + 2*B_row*A_col*B_col = m*r + 2*m*r*r
    matrix_mul_bs2(&denominator_l, H, &denominator);  // cost: B_col*A_col + 2*A_row*A_col*B_col = n*r + 2*r*r*n

    for (int i = 0; i < H->n_row * H->n_col; i++)
        H->M[i] = H->M[i] * numerator.M[i] / denominator.M[i]; // 2*r*n

    for (int count = 0; count < maxIteration; count++) {
        memset(denominator_l, 0, d_rr);
        memset(numerator, 0, d_rn);
        memset(denominator, 0, d_rn);

        memset(numerator_W, 0, d_mr);
        memset(denominator_r, 0, d_rr);

        transpose(H, Ht, m, r);

        ri = mi = ni = 0;
        for (int i = 0; i < r; i += nB) {
            inB = i + nB;
            for (int j = 0; j < n; j += nB) {
                jnB = j + nB;

                // Computation for Wn+1

                // Ht*Ht rmul
                if (j == 0) {
                    ri1 = ri, mi1 = mi;
                    for (int i1 = i; i1 < inB; i1++) {
                        mj1 = 0;
                        for (int j1 = 0; j1 < r; j1 += nB) {
                            for (int k1 = 0; k1 < m; k1 += nB) {
                                mjj1 = mj1;
                                for (int jj1 = j1; jj1 < j1 + nB; jj1++) {
                                    ri1jj1 = ri1 + jj1;
                                    for (int kk1 = k1; kk1 < k1 + nB; kk1++)
                                        denominator_l[ri1jj1] += Ht[mi1 + kk1] * Ht[mjj1 + kk1];
                                    mjj1 += m;
                                }
                            }
                            mj1 += mnB;
                        }
                        ri1 += r;
                        mi1 += m;
                    }
                }

                //V*Ht mul
                mi1 = mi;
                ni1 = ni;
                for (int i1 = i; i1 < inB; i1++) {
                    for (int j1 = j; j1 < jnB; j1++) {
                        ni1j1 = ni1 + j1;
                        for (int k1 = 0; k1 < m; k1++)
                            numerator[ni1j1] += Ht[mi1 + k1] * V[k1 * n + j1];
                    }
                    mi1 += m;
                    ni1 += n;
                }


            }
        }


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
    free(Htmp);
    return err;
}


