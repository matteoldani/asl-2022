#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>

typedef unsigned long long myInt64;

static unsigned int double_size = sizeof(double);

static void transpose(double *src, double *dst, const int N, const int M) {
    //#pragma omp parallel for
    //double *dst = malloc(M * N * sizeof(double));
    for (int n = 0; n < N * M; n++) {
        int i = n / N;
        int j = n % N;
        dst[n] = src[M * j + i];
    }
}

void print_matrix_helper(double *matrix, int n_row, int n_col) {

    printf("Printing a matrix with %d rows and %d cols\n\n", n_row, n_col);
    for (int row = 0; row < n_row; row++) {
        for (int col = 0; col < n_col; col++) {
            fprintf(stdout, "%.2lf\t", matrix[row * n_col + col]);
        }
        fprintf(stdout, "\n\n");
    }
    fprintf(stdout, "\n\n");
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
void matrix_mul_aopt2(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R, int R_n_row,
                      int R_n_col) {
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
void matrix_rtrans_mul_aopt2(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R,
                             int R_n_row, int R_n_col) {

    int Rij;

    for (int i = 0; i < A_n_row; i++) {
        for (int j = 0; j < B_n_row; j++) {
            Rij = i * R_n_col + j;
            R[Rij] = 0;
            for (int k = 0; k < A_n_col; k++) {
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
inline double error(double *approx, double *V, double *W, double *H, int m, int n, int r, int mn, double norm_V) {

    matrix_mul_aopt2(W, m, r, H, r, n, approx, m, n);

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
nnm_factorization_aopt2(double *V_rowM, double *W, double *H, int m, int n, int r, int maxIteration, double epsilon) {
    double *Wt;
    double *H_tmp, *H_switch;
    double *W_tmp, *W_switch;
    double *V_colM;
    int rn, rr, mr, mn;
    rn = r * n;
    rr = r * r;
    mr = m * r;
    mn = m * n;
    Wt = malloc(double_size * mr);
    W_tmp = malloc(double_size * mr);
    H_tmp = malloc(double_size * rn);
    V_colM = malloc(double_size * mn);

    //Operands needed to compute Hn+1
    double *numerator;
    double *denominator_l;
    double *denominator_r;
    double *denominator;    //r x n, r x r, r x n

    numerator = malloc(double_size * rn);
    denominator_l = malloc(double_size * rr);
    denominator_r = malloc(double_size * rr);
    denominator = malloc(double_size * rn);

    //Operands needed to compute Wn+1
    double *numerator_W;
    double *denominator_W;
    double *denominator_l_W;      // m x r, m x r, m x n


    numerator_W = malloc(double_size * mr);
    denominator_W = malloc(double_size * mr);
    denominator_l_W = malloc(double_size * mn);

    double *approximation; //m x n
    approximation = malloc(double_size * mn);


    // this is required to be done here to reuse the same run_opt.
    // does not changhe the number of flops
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            V_colM[j * m + i] = V_rowM[i * n + j];

        }
    }

    double norm_V = 0;
    for (int i = 0; i < mn; i++) {
        norm_V += V_rowM[i] * V_rowM[i];
    }
    norm_V = 1 / sqrt(norm_V);

    //real convergence computation
    double err = -1;
    for (int count = 0; count < maxIteration; count++) {

        err = error(approximation, V_rowM, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }

        transpose(W, Wt, m, r);
        matrix_rtrans_mul_aopt2(Wt, r, m, Wt, r, m, denominator_l, r, r);

        int nij;

        double num_ij, den_ij;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < n; j++) {
                nij = i * n + j;

                num_ij = 0;
                den_ij = 0;
                for (int k = 0; k < m; k++) {
                    num_ij += Wt[i * m + k] * V_colM[j * m + k];
                    if (k < r) {
                        den_ij += denominator_l[i * r + k] * H[k * n + j];
                    }

                }
                H_tmp[nij] = H[nij] * num_ij / den_ij;

            }
        }
        H_switch = H;
        H = H_tmp;
        H_tmp = H_switch;

        matrix_rtrans_mul_aopt2(H, r, n, H, r, n, denominator_r, r, r);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < r; j++) {
                nij = i * r + j;

                num_ij = 0;
                den_ij = 0;
                for (int k = 0; k < n; k++) {
                    num_ij += V_rowM[i * n + k] * H[j * n + k];
                    if (k < r) {
                        den_ij += W[i * r + k] * denominator_r[k * r + j];
                    }
                }
                W_tmp[nij] = W[nij] * num_ij / den_ij;
            }
        }
        W_switch = W;
        W = W_tmp;
        W_tmp = W_switch;

    }

    free(numerator);
    free(denominator);
    free(denominator_l);
    free(denominator_r);
    free(numerator_W);
    free(denominator_W);
    free(denominator_l_W);
    free(Wt);
    free(V_colM);
    free(approximation);
    return err;
}

