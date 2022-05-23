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

    //NEW - introduced blocking and simlified index calcs (code motion, strength reduction)
    int nB = BLOCK_SIZE_TRANS;
    int src_i = 0, src_ii;

    //NEW - introduced double loop to avoid calculating DIV and MOD M*N times
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
matrix_rtrans_mul_opt11(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R, int R_n_row,
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

    // Calculating first iteration of H without interleaving it with W
    transpose(W, Wt, m, r);
    int nij;
    int dij;
    double wt;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < r; i++) {
            nij = i * n + j;
            dij = i * r + j;
            numerator[nij] = 0;
            if (j < r) {
                denominator_l[dij] = 0;
            }
            for (int k = 0; k < m; k++) {
                wt = Wt[i * m + k];
                numerator[nij] += wt * V_colM[j * m + k];
                if (j < r) {
                    denominator_l[dij] += wt * Wt[j * m + k];
                }
            }
        }
    }

    transpose(Wt, W, r, m);
    transpose(H, Ht, r, n);

    int Rij;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < n; j++) {
            Rij = i * n + j;
            denominator[Rij] = 0;
            for (int k = 0; k < r; k++) {
                denominator[Rij] += denominator_l[i * r + k] * Ht[j * r + k];
            }
            Htmp[Rij] = H[Rij] * numerator[Rij] / denominator[Rij];
        }
    }

    tmp = Htmp;
    Htmp = H;
    H = tmp;

    for (int count = 0; count < maxIteration + 1; count++) {

        double h;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < r; j++) {
                Rij = i * r + j;
                numerator_W[Rij] = 0;
                if (i < r) {
                    denominator_l[Rij] = 0;
                }
                for (int k = 0; k < n; k++) {
                    h = H[j * n + k];
                    numerator_W[Rij] += V_rowM[i * n + k] * h;
                    if (i < r) {
                        denominator_l[Rij] += H[i * n + k] * h;
                    }
                }
            }
        }

        int Ri = 0, Rii;
        memset(denominator_W, 0, double_size * mr);
        memset(Wtmp, 0, double_size * mr);
        Rij = 0;
        for (int i = 0; i < m; i += nB) {
            for (int j = 0; j < r; j += nB) {
                for (int k = 0; k < r; k += nB) {
                    Rii = Ri;
                    for (int ii = i; ii < i + nB; ii++) {
                        for (int jj = j; jj < j + nB; jj++) {
                            Rij = Rii + jj;
                            for (int kk = k; kk < k + nB; kk++)
                                denominator_W[Rij] += W[Rii + kk] * denominator_l[kk * r + jj];
                            Wtmp[Rij] = W[Rij] * numerator_W[Rij] / denominator_W[Rij];
                            // TODO: At this point interleave with the next iteration of H
                        }
                        Rii += r;
                    }
                }
            }
            Ri += nB * r;
        }

        tmp = Wtmp;
        Wtmp = W;
        W = tmp;

        // At this point we have one iteration of the algorithm
        err = error(approximation, V_rowM, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }

        if (count == maxIteration) {
            break;
        }

        // Interleaved iteration of H part of the algorithm
        transpose(W, Wt, m, r);
        int nij;
        int dij;
        double wt;

        for (int j = 0; j < n; j++) {
            for (int i = 0; i < r; i++) {
                nij = i * n + j;
                dij = i * r + j;
                numerator[nij] = 0;
                if (j < r) {
                    denominator_l[dij] = 0;
                }
                for (int k = 0; k < m; k++) {
                    wt = Wt[i * m + k];
                    numerator[nij] += wt * V_colM[j * m + k];
                    if (j < r) {
                        denominator_l[dij] += wt * Wt[j * m + k];
                    }
                }
            }
        }

        transpose(Wt, W, r, m);
        transpose(H, Ht, r, n);

        int Rij;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < n; j++) {
                Rij = i * n + j;
                denominator[Rij] = 0;
                for (int k = 0; k < r; k++) {
                    denominator[Rij] += denominator_l[i * r + k] * Ht[j * r + k];
                }
                Htmp[Rij] = H[Rij] * numerator[Rij] / denominator[Rij];
            }
        }

        tmp = Htmp;
        Htmp = H;
        H = tmp;

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


