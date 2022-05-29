#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <optimizations/optimizations_44.h>
#include <immintrin.h>
//NEW from opt 24 call to blas substituted to call to matrix mul from 43

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

// TODO write the version with j - i order to use for computation of Hn+1
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
void matrix_mul_opt44(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R, int R_n_row, int R_n_col)
{

    int Rij = 0, Ri = 0, Ai = 0, Aii, Rii;
    int nB = BLOCK_SIZE_MMUL;
    int nBR_n_col = nB * R_n_col;
    int nBA_n_col = nB * A_n_col;
    int unroll_i = 2, unroll_j = 16;
    int kk, i, j, k;

    __m256d a0, a1;
    __m256d b0, b1, b2, b3;
    __m256d r0, r1, r2, r3;
    __m256d r4, r5, r6, r7;



    memset(R, 0, double_size * R_n_row * R_n_col);

    for (i = 0; i < A_n_row - nB + 1; i += nB)
    {
        for (j = 0; j < B_n_col - nB + 1; j += nB)
        {
            for (k = 0; k < A_n_col - nB + 1; k += nB)
            {
                Rii = Ri;
                Aii = Ai;
                for (int ii = i; ii < i + nB - unroll_i + 1; ii += unroll_i)
                {

                    for (int jj = j; jj < j + nB - unroll_j + 1; jj += unroll_j)
                    {
                        
                        Rij = Rii + jj;
                        int idx_r = Rij + R_n_col;
                        
                        r0 = _mm256_loadu_pd((double *)&R[Rij]);
                        r1 = _mm256_loadu_pd((double *)&R[Rij + 4]);
                        r2 = _mm256_loadu_pd((double *)&R[Rij + 8]);
                        r3 = _mm256_loadu_pd((double *)&R[Rij + 12]);

                        r4 = _mm256_loadu_pd((double *)&R[idx_r]);
                        r5 = _mm256_loadu_pd((double *)&R[idx_r + 4]);
                        r6 = _mm256_loadu_pd((double *)&R[idx_r + 8]);
                        r7 = _mm256_loadu_pd((double *)&R[idx_r + 12]);


                        int idx_b = k*B_n_col + jj;
                        for (kk = k; kk < k + nB; kk++)
                        {
                            // printf("ii:%d, jj:%d k:%d kk: %d\n",ii, jj, k, kk);
                            a0 = _mm256_set1_pd(A[Aii + kk]);                //Aik0 = A[Aii + kk];
                            a1 = _mm256_set1_pd(A[Aii + A_n_col + kk]);      //Aik1 = A[Aii + A_n_col + kk]; 
                            
                            b0 = _mm256_loadu_pd((double *)&B[idx_b]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b1 = _mm256_loadu_pd((double *)&B[idx_b + 4]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b2 = _mm256_loadu_pd((double *)&B[idx_b + 8]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b3 = _mm256_loadu_pd((double *)&B[idx_b + 12]);    // Bi0j0 = B[kk * B_n_col + jj];
  
                            r0 = _mm256_fmadd_pd(a0, b0, r0);
                            r1 = _mm256_fmadd_pd(a0, b1, r1);
                            r2 = _mm256_fmadd_pd(a0, b2, r2);
                            r3 = _mm256_fmadd_pd(a0, b3, r3);

                            r4 = _mm256_fmadd_pd(a1, b0, r4);
                            r5 = _mm256_fmadd_pd(a1, b1, r5);
                            r6 = _mm256_fmadd_pd(a1, b2, r6);
                            r7 = _mm256_fmadd_pd(a1, b3, r7);

                            idx_b += B_n_col;
                        }

                        // _mm256_storeu_pd((double *)&R[Rij], r0);
                        // _mm256_storeu_pd((double *)&R[Rij + R_n_col], r1);

                        _mm256_storeu_pd((double *)&R[Rij], r0);
                        _mm256_storeu_pd((double *)&R[Rij + 4], r1);
                        _mm256_storeu_pd((double *)&R[Rij + 8], r2);
                        _mm256_storeu_pd((double *)&R[Rij + 12], r3);

                        _mm256_storeu_pd((double *)&R[idx_r], r4);
                        _mm256_storeu_pd((double *)&R[idx_r + 4], r5);
                        _mm256_storeu_pd((double *)&R[idx_r + 8], r6);
                        _mm256_storeu_pd((double *)&R[idx_r + 12], r7);


                        for (; kk < k + nB; kk++)
                        {
                            R[Rij] += A[Aii + kk] * B[kk * B_n_col + jj];
                        }
                    }
                    Rii += R_n_col * unroll_i;
                    Aii += A_n_col * unroll_i;
                }
            }
            //// clean up
            for (int ii = i; ii < i + nB; ii++)
            {
                // printf("Clean up on blocks\n");
                for (int jj = j; jj < j + nB; jj++)
                {
                    Rij = Rii + jj;
                    //R_Rij0 = 0;
                    for (kk = k; kk < A_n_col; kk++)
                    {
                        // printf("ii:%d jj:%d kk%d\n", ii, jj, kk);
                        R[ii * B_n_col + jj] += A[ii * A_n_col + kk] * B[kk * B_n_col + jj];
                    }
                    // printf("\n");
                }
            }
            //// end clean up
        }
        //// clean up
        for (int ii = i; ii < i + nB; ii++)
        {

            for (int jj = j; jj < B_n_col; jj++)
            {
                for (kk = 0; kk < A_n_col; kk++)
                {
                    // printf("ii:%d jj:%d kk%d\n", ii, jj, kk);
                    R[ii * B_n_col + jj] += A[ii * A_n_col + kk] * B[kk * B_n_col + jj];
                }
            }
        }
        //// end clean up

        Ri += nBR_n_col;
        Ai += nBA_n_col;
    }
    //// clean up
    for (; i < A_n_row; i++)
    {
        for (int j = 0; j < B_n_col; j++)
        {
            for (k = 0; k < A_n_col; k++)
            {
                R[i * B_n_col + j] += A[i * A_n_col + k] * B[k * B_n_col + j];
            }
        }
    }
    //// end clean up
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

    matrix_mul_opt44(W, m, r, H, r, n, approx, m, n);

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
double nnm_factorization_opt44(double *V, double*W, double*H, int m, int n, int r, int maxIteration, double epsilon) {

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
        norm_V1 += V[i]   * V[i];
        norm_V2 += V[i+1] * V[i+1];
        norm_V3 += V[i+2] * V[i+2];
        norm_V4 += V[i+3] * V[i+3];
        norm_V5 += V[i+4] * V[i+4];
        norm_V6 += V[i+5] * V[i+5];
        norm_V7 += V[i+6] * V[i+6];
        norm_V8 += V[i+7] * V[i+7];
    }

    norm_V = norm_V1 + norm_V2 + norm_V3 + norm_V4 + norm_V5 + norm_V6 + norm_V7 + norm_V8;

    for (; i < mn; i++)
    {
        norm_V += V[i] * V[i];
    }

    norm_V = 1 / sqrt(norm_V);

    //real convergence computation
    double err = -1;	

    double *Wt = malloc(double_size * mr);
    double *Ht = malloc(double_size * rn);		

    for (int count = 0; count < maxIteration; count++) {
     
        err = error(approximation, V, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }

        //computation for Hn+1
        
        transpose(W, Wt, m, r);
        matrix_mul_opt44(Wt, r, m, V, m, n, numerator, r, n);
        matrix_mul_opt44(Wt, r, m, W, m, r, denominator_l, r, r);
        // matrix_ltrans_mul_opt24(W, m, r, V, m, n, numerator, r, n);
        // matrix_ltrans_mul_opt24(W, m, r, W, m, r, denominator_l, r, r);
        matrix_mul_opt44(denominator_l, r, r, H, r, n, denominator, r, n);
 

        idx_unroll = rn/8;
        for (i = 0; i < idx_unroll; i+=8){
            H[i] = H[i] * numerator[i] / denominator[i];
            H[i+1] = H[i+1] * numerator[i+1] / denominator[i+1];
            H[i+2] = H[i+2] * numerator[i+2] / denominator[i+2];
            H[i+3] = H[i+3] * numerator[i+3] / denominator[i+3];
            H[i+4] = H[i+4] * numerator[i+4] / denominator[i+4];
            H[i+5] = H[i+5] * numerator[i+5] / denominator[i+5];
            H[i+6] = H[i+6] * numerator[i+6] / denominator[i+6];
            H[i+7] = H[i+7] * numerator[i+7] / denominator[i+7];
        }
        for(;i<rn;i++){
            H[i] = H[i] * numerator[i] / denominator[i];
        }

        //computation for Wn+1
        double *Ht = malloc(double_size * rn);
        transpose(H, Ht, r, n);
        matrix_mul_opt44(V, m, n, Ht, n, r, numerator_W, m, r);
        matrix_mul_opt44(H, r, n, Ht, n, r, denominator_l, r, r);
        matrix_mul_opt44(W, m, r, denominator_l, r, r, denominator_W, m, r);
        // matrix_rtrans_mul_opt24(V, m, n, H, r, n, numerator_W, m, r);
        // matrix_rtrans_mul_opt24(H, r, n, H, r, n, denominator_l, r, r);
        // matrix_mul_opt24(W, m, r, denominator_l, r, r, denominator_W, m, r);

        idx_unroll = mr / 8;
        for (i = 0; i < idx_unroll; i+=8){
            W[i] = W[i] * numerator_W[i] / denominator_W[i];
            W[i+1] = W[i+1] * numerator_W[i+1] / denominator_W[i+1];
            W[i+2] = W[i+2] * numerator_W[i+2] / denominator_W[i+2];
            W[i+3] = W[i+3] * numerator_W[i+3] / denominator_W[i+3];
            W[i+4] = W[i+4] * numerator_W[i+4] / denominator_W[i+4];
            W[i+5] = W[i+5] * numerator_W[i+5] / denominator_W[i+5];
            W[i+6] = W[i+6] * numerator_W[i+6] / denominator_W[i+6];
            W[i+7] = W[i+7] * numerator_W[i+7] / denominator_W[i+7];
        }
        for(;i<mr;i++){
            W[i] = W[i] * numerator_W[i] / denominator_W[i];
        }
    }

    free(numerator);
    free(denominator);
    free(denominator_l);
    free(numerator_W);
    free(denominator_W);
    free(approximation);
    free(Wt);
    free(Ht);
    return err;
}
