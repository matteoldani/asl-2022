#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <optimizations/optimizations_41.h>

#include "cblas.h"

//NEW - optimization done on optimization_24 using dsyrk

static unsigned int double_size = sizeof(double);

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
void matrix_mul_opt41(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R, int R_n_row,
                      int R_n_col) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A_n_row, B_n_col, A_n_col, 1,
                A, A_n_col, B, B_n_col,
                0, R, B_n_col);
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
void matrix_ltrans_mul_opt41(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R,
                             int R_n_row, int R_n_col) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                A_n_col, B_n_col, B_n_row, 1, //r=A->n_row = B->n_row
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
void matrix_rtrans_mul_opt41(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R,
                             int R_n_row, int R_n_col) {
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
inline double error(double *approx, double *V, double *W, double *H, int m, int n, int r, int mn, double norm_V) {

    matrix_mul_opt41(W, m, r, H, r, n, approx, m, n);

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

    int idx_unroll = mn / 8;
    int i;
    for (i = 0; i < idx_unroll; i += 8) {
        temp1 = V[i] - approx[i];
        temp2 = V[i + 1] - approx[i + 1];
        temp3 = V[i + 2] - approx[i + 2];
        temp4 = V[i + 3] - approx[i + 3];
        temp5 = V[i + 4] - approx[i + 4];
        temp6 = V[i + 5] - approx[i + 5];
        temp7 = V[i + 6] - approx[i + 6];
        temp8 = V[i + 7] - approx[i + 7];

        norm_approx1 += temp1 * temp1;
        norm_approx2 += temp2 * temp2;
        norm_approx3 += temp3 * temp3;
        norm_approx4 += temp4 * temp4;
        norm_approx5 += temp5 * temp5;
        norm_approx6 += temp6 * temp6;
        norm_approx7 += temp7 * temp7;
        norm_approx8 += temp8 * temp8;

    }

    norm_approx =
            norm_approx1 + norm_approx2 + norm_approx3 + norm_approx4 + norm_approx5 + norm_approx6 + norm_approx7 +
            norm_approx8;

    for (; i < mn; i++) {
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
double nnm_factorization_opt41(double *V, double *W, double *H, int m, int n, int r, int maxIteration, double epsilon) {

    int rn, rr, mr, mn;
    rn = r * n;
    rr = r * r;
    mr = m * r;
    mn = m * n;

    //Operands needed to compute Hn+1
    double *numerator, *denominator_l, *denominator;    //r x n, r x r, r x n
    numerator = malloc(double_size * rn);
    denominator_l = malloc(double_size * rr);
    denominator = malloc(double_size * rn);

    //Operands needed to compute Wn+1
    double *numerator_W, *denominator_W;      // m x r, m x r, m x n
    numerator_W = malloc(double_size * mr);
    denominator_W = malloc(double_size * mr);


    double *approximation; //m x n
    approximation = malloc(double_size * mn);

    double norm_V = 0;
    double norm_V1 = 0;
    double norm_V2 = 0;
    double norm_V3 = 0;
    double norm_V4 = 0;
    double norm_V5 = 0;
    double norm_V6 = 0;
    double norm_V7 = 0;
    double norm_V8 = 0;


    int idx_unroll = mn / 8;
    int i;

    ///// NORM

    for (i = 0; i < idx_unroll; i += 8) {
        norm_V1 += V[i] * V[i];
        norm_V2 += V[i + 1] * V[i + 1];
        norm_V3 += V[i + 2] * V[i + 2];
        norm_V4 += V[i + 3] * V[i + 3];
        norm_V5 += V[i + 4] * V[i + 4];
        norm_V6 += V[i + 5] * V[i + 5];
        norm_V7 += V[i + 6] * V[i + 6];
        norm_V8 += V[i + 7] * V[i + 7];
    }

    norm_V = norm_V1 + norm_V2 + norm_V3 + norm_V4 + norm_V5 + norm_V6 + norm_V7 + norm_V8;

    for (; i < mn; i++) {
        norm_V += V[i] * V[i];
    }

    norm_V = 1 / sqrt(norm_V);

    //real convergence computation
    double err = -1;
    for (int count = 0; count < maxIteration; count++) {

        err = error(approximation, V, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }


        matrix_ltrans_mul_opt41(W, m, r, V, m, n, numerator, r, n);
        //NEW matrix mult substituted with dsyrk and dsymm to exploit matrix symmetry
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, r, m, 1.0, W, r, 0.0, denominator_l, r);
        cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, r, n, 1.0, denominator_l, r, H, n, 0.0, denominator, n);


        idx_unroll = rn / 8;
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


        //computation for Wn+1
        matrix_rtrans_mul_opt41(V, m, n, H, r, n, numerator_W, m, r);
        //NEW matrix mult substituted with dsyrk and dsymm to exploit matrix symmetry
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, r, n, 1.0, H, n, 0.0, denominator_l, r);
        cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper, m, r, 1.0, denominator_l, r, W, r, 0.0, denominator_W, r);

        idx_unroll = mr / 8;
        for (i = 0; i < idx_unroll; i += 8) {
            W[i] = W[i] * numerator_W[i] / denominator_W[i];
            W[i + 1] = W[i + 1] * numerator_W[i + 1] / denominator_W[i + 1];
            W[i + 2] = W[i + 2] * numerator_W[i + 2] / denominator_W[i + 2];
            W[i + 3] = W[i + 3] * numerator_W[i + 3] / denominator_W[i + 3];
            W[i + 4] = W[i + 4] * numerator_W[i + 4] / denominator_W[i + 4];
            W[i + 5] = W[i + 5] * numerator_W[i + 5] / denominator_W[i + 5];
            W[i + 6] = W[i + 6] * numerator_W[i + 6] / denominator_W[i + 6];
            W[i + 7] = W[i + 7] * numerator_W[i + 7] / denominator_W[i + 7];
        }
        for (; i < mr; i++) {
            W[i] = W[i] * numerator_W[i] / denominator_W[i];
        }
    }

    free(numerator);
    free(denominator);
    free(denominator_l);
    free(numerator_W);
    free(denominator_W);
    free(approximation);
    return err;
}
