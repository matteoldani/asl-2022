#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <optimizations/optimizations_53.h>
#include <immintrin.h>

//NEW - on top of opt_47 for the base and opt_35 for the algorithmic optimization
//NEW - this implementation is generalized - it can run with any m and n

typedef unsigned long long myInt64;

static unsigned int double_size = sizeof(double);

inline void transpose4x4(double* dst, double* src, const int n, const int m) {

    __m256d tmp0, tmp1, tmp2, tmp3;
    __m256d row0, row1, row2, row3;

    row0 = _mm256_loadu_pd(&src[0 * m]);
    row1 = _mm256_loadu_pd(&src[1 * m]);
    row2 = _mm256_loadu_pd(&src[2 * m]);
    row3 = _mm256_loadu_pd(&src[3 * m]);

    tmp0 = _mm256_shuffle_pd(row0, row1, 0x0);
    tmp2 = _mm256_shuffle_pd(row0, row1, 0xF);
    tmp1 = _mm256_shuffle_pd(row2, row3, 0x0);
    tmp3 = _mm256_shuffle_pd(row2, row3, 0xF);

    row0 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
    row1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
    row2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
    row3 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

    _mm256_storeu_pd(&dst[0 * n], row0);
    _mm256_storeu_pd(&dst[1 * n], row1);
    _mm256_storeu_pd(&dst[2 * n], row2);
    _mm256_storeu_pd(&dst[3 * n], row3);
}

static void transpose(double* src, double* dst, const int n, const int m) {

    int nB = BLOCK_SIZE_TRANS;

    int i, j, i2, j2;
    int n_nB = n - nB, m_nB = m - nB, inB, jnB, m_4 = m - 4, n_4 = n - 4;

    for (i = 0; i <= n_nB; i += nB) {
        inB = i + nB;
        for (j = 0; j <= m_nB; j += nB) {
            jnB = j + nB;
            for (i2 = i; i2 < inB; i2 += 4) {
                for (j2 = j; j2 < jnB; j2 += 4) {
                    transpose4x4(&dst[j2 * n + i2], &src[i2 * m + j2], n, m);
                }
            }
        }
        //if number of columns is not divisible by block size
        if (j != m) {
            for (i2 = i; i2 < inB; i2 += 4) {
                for (j2 = j; j2 <= m_4; j2 += 4)
                    transpose4x4(&dst[j2 * n + i2], &src[i2 * m + j2], n, m);
                //if number of columns is not divisible by 4
                for (; j2 < m; j2++) {
                    dst[j2 * n + i2] = src[i2 * m + j2];
                    dst[j2 * n + i2 + 1] = src[(i2 + 1) * m + j2];
                    dst[j2 * n + i2 + 2] = src[(i2 + 2) * m + j2];
                    dst[j2 * n + i2 + 3] = src[(i2 + 3) * m + j2];
                }
            }
        }
    }
    //if number of rows is not divisible by block size
    for (; i <= n_4; i += 4) {
        for (j = 0; j < m_4; j += 4)
            transpose4x4(&dst[j * n + i], &src[i * m + j], n, m);
        //if number of columns is not divisible by 4
        for (; j < m; j++) {
            dst[j * n + i] = src[i * m + j];
            dst[j * n + i + 1] = src[(i + 1) * m + j];
            dst[j * n + i + 2] = src[(i + 2) * m + j];
            dst[j * n + i + 3] = src[(i + 3) * m + j];
        }
    }
    //if number of rows is not divisible by 4
    for (; i < n; i++)
        for (j = 0; j < m; j++)
            dst[j * n + i] = src[i * m + j];
}

static void pad_matrix(double** M, int* r, int* c) {
    int temp_r;
    int temp_c;

    if (((*r) % BLOCK_SIZE_MMUL == 0) && ((*c) % BLOCK_SIZE_MMUL == 0)) {
        return;
    }

    if ((*r) % BLOCK_SIZE_MMUL != 0) {
        temp_r = (((*r) / BLOCK_SIZE_MMUL) + 1) * BLOCK_SIZE_MMUL;
    }
    else {
        temp_r = *r;
    }

    if ((*c) % BLOCK_SIZE_MMUL != 0) {
        temp_c = (((*c) / BLOCK_SIZE_MMUL) + 1) * BLOCK_SIZE_MMUL;
    }
    else {
        temp_c = *c;
    }

    double* new_Mt;

    *M = realloc(*M, double_size * (*c) * temp_r);
    // i need to pad the rows before and the cols after transposing
    memset(&(*M)[(*c) * (*r)], 0, double_size * (temp_r - (*r)) * (*c));

    new_Mt = aligned_alloc(32, double_size * temp_c * temp_r);
    transpose(*M, new_Mt, temp_r, *c);
    memset(&new_Mt[temp_r * (*c)], 0, double_size * (temp_c - (*c)) * temp_r);

    free(*M);
    *M = aligned_alloc(32, double_size * temp_c * temp_r);
    *c = temp_c;
    *r = temp_r;
    transpose(new_Mt, *M, temp_c, temp_r);

    free(new_Mt);
}

static void unpad_matrix(double** M, int* r, int* c, int original_r, int original_c) {

    // lets suppose that are always row majour

    // i can remove the last useless rows
    *M = realloc(*M, (*c) * original_r * double_size);

    // i need to transpose and remove the rest
    double* new_Mt = aligned_alloc(32, (*c) * original_r * double_size);
    transpose(*M, new_Mt, original_r, *c);

    // i need to resize the transoposed
    new_Mt = realloc(new_Mt, double_size * original_c * original_r);

    // ie need to transpose back
    free(*M);
    *M = aligned_alloc(32, double_size * original_c * original_r);
    transpose(new_Mt, *M, original_c, original_r);

    *r = original_r;
    *c = original_c;

    free(new_Mt);
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
void matrix_mul_opt53_padding(double* A_final, int A_n_row, int A_n_col, double* B_final, int B_n_row, int B_n_col, double* R_final, int R_n_row, int R_n_col) {

    int m = A_n_row;
    int n = B_n_col;
    int r = B_n_row;
    double* V, * H, * W;
    V = aligned_alloc(32, double_size * m * n);
    H = aligned_alloc(32, double_size * r * n);
    W = aligned_alloc(32, double_size * m * r);

    memcpy(V, R_final, m * n * double_size);
    memcpy(W, A_final, m * r * double_size);
    memcpy(H, B_final, r * n * double_size);

    // padding all the values to multiple of BLOCKSIZE
    int temp_r = r;
    int temp_m = m;
    int temp_n = n;

    int original_m = m;
    int original_n = n;
    int original_r = r;

    // i do not have to modify m n yet
    pad_matrix(&V, &temp_m, &temp_n);
    // i do not have to modify r but i can modify m
    pad_matrix(&W, &m, &temp_r);
    // i can modify both r and n
    pad_matrix(&H, &r, &n);

    matrix_mul_opt53(W, m, r, H, r, m, V, m, n);

    unpad_matrix(&V, &temp_m, &temp_n, original_m, original_n);
    unpad_matrix(&W, &m, &temp_r, original_m, original_r);
    unpad_matrix(&H, &r, &n, original_r, original_n);

    memcpy(R_final, V, m * n * double_size);
    memcpy(A_final, W, m * r * double_size);
    memcpy(B_final, H, r * n * double_size);

    free(V);
    free(H);
    free(W);
}


// NOTE a possible improvement is to  write the version with j - i order to use for computation of Hn+1
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
void matrix_mul_opt53(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col) {

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
    //MAIN LOOP BLOCKED 16x16
    for (i = 0; i < A_n_row - nB + 1; i += nB) {
        for (j = 0; j < B_n_col - nB + 1; j += nB) {
            for (k = 0; k < A_n_col - nB + 1; k += nB) {

                Rii = Ri;
                Aii = Ai;
                for (int ii = i; ii < i + nB - unroll_i + 1; ii += unroll_i) {
                    for (int jj = j; jj < j + nB - unroll_j + 1; jj += unroll_j) {

                        Rij = Rii + jj;
                        int idx_r = Rij + R_n_col;

                        r0 = _mm256_loadu_pd((double*)&R[Rij]);
                        r1 = _mm256_loadu_pd((double*)&R[Rij + 4]);
                        r2 = _mm256_loadu_pd((double*)&R[Rij + 8]);
                        r3 = _mm256_loadu_pd((double*)&R[Rij + 12]);

                        r4 = _mm256_loadu_pd((double*)&R[idx_r]);
                        r5 = _mm256_loadu_pd((double*)&R[idx_r + 4]);
                        r6 = _mm256_loadu_pd((double*)&R[idx_r + 8]);
                        r7 = _mm256_loadu_pd((double*)&R[idx_r + 12]);

                        int idx_b = k * B_n_col + jj;
                        for (kk = k; kk < k + nB; kk++) {
                            a0 = _mm256_set1_pd(A[Aii + kk]);                //Aik0 = A[Aii + kk];
                            a1 = _mm256_set1_pd(A[Aii + A_n_col + kk]);      //Aik1 = A[Aii + A_n_col + kk]; 

                            b0 = _mm256_loadu_pd((double*)&B[idx_b]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b1 = _mm256_loadu_pd((double*)&B[idx_b + 4]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b2 = _mm256_loadu_pd((double*)&B[idx_b + 8]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b3 = _mm256_loadu_pd((double*)&B[idx_b + 12]);    // Bi0j0 = B[kk * B_n_col + jj];

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

                        _mm256_storeu_pd((double*)&R[Rij], r0);
                        _mm256_storeu_pd((double*)&R[Rij + 4], r1);
                        _mm256_storeu_pd((double*)&R[Rij + 8], r2);
                        _mm256_storeu_pd((double*)&R[Rij + 12], r3);

                        _mm256_storeu_pd((double*)&R[idx_r], r4);
                        _mm256_storeu_pd((double*)&R[idx_r + 4], r5);
                        _mm256_storeu_pd((double*)&R[idx_r + 8], r6);
                        _mm256_storeu_pd((double*)&R[idx_r + 12], r7);
                    }
                    Rii += R_n_col * unroll_i;
                    Aii += A_n_col * unroll_i;
                }
            }
        }
        Ri += nBR_n_col;
        Ai += nBA_n_col;
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
static inline double error(double* approx, double* V, double* W, double* H, int m, int n, int r, int mn, double norm_V) {

    matrix_mul_opt53(W, m, r, H, r, n, approx, m, n);

    double* norm;
    double res;

    norm = aligned_alloc(32, double_size * 4);

    __m256d norm_approx0;
    __m256d norm_approx1;
    __m256d norm_approx2;
    __m256d norm_approx3;

    __m256d t;

    __m256d r0, r1, r2, r3;
    __m256d r4, r5, r6, r7;
    __m256d t0, t1, t2, t3;

    norm_approx0 = _mm256_setzero_pd();
    norm_approx1 = _mm256_setzero_pd();
    norm_approx2 = _mm256_setzero_pd();
    norm_approx3 = _mm256_setzero_pd();

    int i;
    for (i = 0; i < mn; i += 16) {

        r0 = _mm256_loadu_pd((double*)&V[i]);
        r1 = _mm256_loadu_pd((double*)&V[i + 4]);
        r2 = _mm256_loadu_pd((double*)&V[i + 8]);
        r3 = _mm256_loadu_pd((double*)&V[i + 12]);

        r4 = _mm256_loadu_pd((double*)&approx[i]);
        r5 = _mm256_loadu_pd((double*)&approx[i + 4]);
        r6 = _mm256_loadu_pd((double*)&approx[i + 8]);
        r7 = _mm256_loadu_pd((double*)&approx[i + 12]);

        t0 = _mm256_sub_pd(r0, r4);
        t1 = _mm256_sub_pd(r1, r5);
        t2 = _mm256_sub_pd(r2, r6);
        t3 = _mm256_sub_pd(r3, r7);

        norm_approx0 = _mm256_fmadd_pd(t0, t0, norm_approx0);
        norm_approx1 = _mm256_fmadd_pd(t1, t1, norm_approx1);
        norm_approx2 = _mm256_fmadd_pd(t2, t2, norm_approx2);
        norm_approx3 = _mm256_fmadd_pd(t3, t3, norm_approx3);
    }

    norm_approx0 = _mm256_add_pd(norm_approx0, norm_approx1);
    norm_approx2 = _mm256_add_pd(norm_approx2, norm_approx3);
    norm_approx0 = _mm256_add_pd(norm_approx0, norm_approx2);
    t = _mm256_hadd_pd(norm_approx0, norm_approx0);
    _mm256_storeu_pd(&norm[0], t);
    res = sqrt(norm[0] + norm[2]);
    return res * norm_V;
}

static inline int min(int a, int b) {
    return a < b ? a : b;
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
double nnm_factorization_opt53(double* V_final, double* W_final, double* H_final, int m, int n, int r, int maxIteration, double epsilon) {

    double* V, * W, * H;

    V = aligned_alloc(32, double_size * m * n);
    H = aligned_alloc(32, double_size * r * n);
    W = aligned_alloc(32, double_size * m * r);

    memcpy(V, V_final, m * n * double_size);
    memcpy(W, W_final, m * r * double_size);
    memcpy(H, H_final, r * n * double_size);

    // padding all the values to multiple of BLOCKSIZE
    int temp_r = r;
    int temp_m = m;
    int temp_n = n;

    int original_m = m;
    int original_n = n;
    int original_r = r;

    // i do not have to modify m n yet
    pad_matrix(&V, &temp_m, &temp_n);
    // i do not have to modify r but i can modify m
    pad_matrix(&W, &m, &temp_r);
    // i can modify both r and n
    pad_matrix(&H, &r, &n);

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

    //Operands needed to compute Hn+1
    double* numerator;      //r x n
    double* denominator_l;  //r x r
    double* denominator;    //r x n

    numerator = aligned_alloc(32, d_rn);
    denominator_l = aligned_alloc(32, d_rr);
    denominator = aligned_alloc(32, d_rn);

    //Operands needed to compute Wn+1
    double* numerator_W;    //m x r
    double* denominator_r;  //r x r
    double* denominator_W;  //m x r

    numerator_W = aligned_alloc(32, d_mr);
    denominator_r = aligned_alloc(32, d_rr);
    denominator_W = aligned_alloc(32, d_mr);

    double* approximation; //m x n
    approximation = aligned_alloc(32, d_mn);

    double norm_V = 0;
    double* norm_tmp = aligned_alloc(32, double_size * 4);
    int i;

    __m256d norm_approx0, norm_approx1, norm_approx2, norm_approx3;
    __m256d t;

    __m256d r0, r1, r2, r3;

    __m256d sum0, sum1;

    norm_approx0 = _mm256_setzero_pd();
    norm_approx1 = _mm256_setzero_pd();
    norm_approx2 = _mm256_setzero_pd();
    norm_approx3 = _mm256_setzero_pd();

    for (i = 0; i < mn; i += 16) {

        r0 = _mm256_loadu_pd((double*)&V[i]);
        r1 = _mm256_loadu_pd((double*)&V[i + 4]);
        r2 = _mm256_loadu_pd((double*)&V[i + 8]);
        r3 = _mm256_loadu_pd((double*)&V[i + 12]);

        norm_approx0 = _mm256_fmadd_pd(r0, r0, norm_approx0);
        norm_approx1 = _mm256_fmadd_pd(r1, r1, norm_approx1);
        norm_approx2 = _mm256_fmadd_pd(r2, r2, norm_approx2);
        norm_approx3 = _mm256_fmadd_pd(r3, r3, norm_approx3);
    }


    sum0 = _mm256_add_pd(norm_approx0, norm_approx1);
    sum1 = _mm256_add_pd(norm_approx2, norm_approx3);
    sum0 = _mm256_add_pd(sum0, sum1);

    t = _mm256_hadd_pd(sum0, sum0);
    _mm256_storeu_pd(&norm_tmp[0], t);
    norm_V = 1 / sqrt(norm_tmp[0] + norm_tmp[2]);

    double *Wt, *W_new, *Ht;

    Wt = aligned_alloc(32, d_mr);
    Ht = aligned_alloc(32, d_rn);
    W_new = aligned_alloc(32, d_mr);

    int nB_i = BLOCK_SIZE_W_ROW;
    int nB_j = BLOCK_SIZE_W_COL;
    int rnB_i = r * nB_i, nnB_i = n * nB_i, mnB_i = m * nB_i;
    int rnB_j = r * nB_j, nnB_j = n * nB_j, mnB_j = m * nB_j;
    int inB, jnB, n_2 = n << 1, r_2 = r << 1, m_2 = m << 1;; 
    int ri, ni, mi, rj, nj, mj, ri1, ni1, ni1j1, ri1j1, idx_r, idx_b, mi1;

    //Precompute first parts of H so we can start with calculating W blockwise and reusing blocks for next H
    //pre-computation for H1
    transpose(W, Wt, m, r);
    matrix_mul_opt53(Wt, r, m, V, m, n, numerator, r, n);
    matrix_mul_opt53(Wt, r, m, W, m, r, denominator_l, r, r);

    __m256d num, fac, den, res;
    __m256d a0, a1;
    __m256d b0, b1, b2, b3;
    __m256d r4, r5, r6, r7;

    //real convergence computation
    double err = -1;
    for (int count = 0; count < maxIteration; count++) {

        err = error(approximation, V, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }

        //remaining computation for Hn+1
        
        matrix_mul_opt53(denominator_l, r, r, H, r, n, denominator, r, n);

        for (i = 0; i < original_r; i++) {
            for (int j = 0; j < original_n; j++) {
                H[i * n + j] = H[i * n + j] * numerator[i * n + j] / denominator[i * n + j];
            }
        }
        
       
        //computation for Wn+1  
        
        memset(numerator, 0, d_rn);
        memset(denominator_l, 0, d_rr);

        transpose(H, Ht, r, n);
        
        //Since we need a column of HHt per block of W we would have to calculate all of HHt while calculating the first row of blocks of W, so it's better to calculate it in advance
        matrix_mul_opt53(H, r, n, Ht, n, r, denominator_r, r, r);

        //NEW - WE calculate W block by block and reuse it instantly for Wt*V and Wt*W
        //NEW - All operations done on blocks are now done optimally - using vector instructions
        ri = ni = mi = 0;
        for (int i = 0; i < m; i += nB_i) {
            inB = i + nB_i;
            nj = 0;
            rj = 0;
            mj = 0;
            for (int j = 0; j < r; j += nB_j) {
                jnB = j + nB_j;

                //computation for Wn+1

                //VHt mul
                ni1 = ni;
                ri1 = ri;
                for (int i1 = i; i1 <= inB - 2; i1 += 2) {
                    for (int j1 = j; j1 <= jnB - 16; j1 += 16) {
                        ri1j1 = ri1 + j1;
                        idx_r = ri1j1 + r;

                        r0 = _mm256_setzero_pd();
                        r1 = _mm256_setzero_pd();
                        r2 = _mm256_setzero_pd();
                        r3 = _mm256_setzero_pd();

                        r4 = _mm256_setzero_pd();
                        r5 = _mm256_setzero_pd();
                        r6 = _mm256_setzero_pd();
                        r7 = _mm256_setzero_pd();

                        idx_b = j1;
                        for (int k1 = 0; k1 < n; k1++) {
                            a0 = _mm256_set1_pd(V[ni1 + k1]);
                            a1 = _mm256_set1_pd(V[ni1 + n + k1]);

                            b0 = _mm256_load_pd(&Ht[idx_b]);
                            b1 = _mm256_load_pd(&Ht[idx_b + 4]);
                            b2 = _mm256_load_pd(&Ht[idx_b + 8]);
                            b3 = _mm256_load_pd(&Ht[idx_b + 12]);

                            r0 = _mm256_fmadd_pd(a0, b0, r0);
                            r1 = _mm256_fmadd_pd(a0, b1, r1);
                            r2 = _mm256_fmadd_pd(a0, b2, r2);
                            r3 = _mm256_fmadd_pd(a0, b3, r3);

                            r4 = _mm256_fmadd_pd(a1, b0, r4);
                            r5 = _mm256_fmadd_pd(a1, b1, r5);
                            r6 = _mm256_fmadd_pd(a1, b2, r6);
                            r7 = _mm256_fmadd_pd(a1, b3, r7);

                            idx_b += r;
                        }

                        _mm256_store_pd(&numerator_W[ri1j1], r0);
                        _mm256_store_pd(&numerator_W[ri1j1 + 4], r1);
                        _mm256_store_pd(&numerator_W[ri1j1 + 8], r2);
                        _mm256_store_pd(&numerator_W[ri1j1 + 12], r3);

                        _mm256_store_pd(&numerator_W[idx_r], r4);
                        _mm256_store_pd(&numerator_W[idx_r + 4], r5);
                        _mm256_store_pd(&numerator_W[idx_r + 8], r6);
                        _mm256_store_pd(&numerator_W[idx_r + 12], r7);
                    }
                    ni1 += n_2;
                    ri1 += r_2;
                }

                //W(HHt) mul
                ri1 = ri;
                for (int i1 = i; i1 <= inB - 2; i1 += 2) {
                    for (int j1 = j; j1 <= jnB - 16; j1 += 16) {
                        ri1j1 = ri1 + j1;
                        idx_r = ri1j1 + r;

                        r0 = _mm256_setzero_pd();
                        r1 = _mm256_setzero_pd();
                        r2 = _mm256_setzero_pd();
                        r3 = _mm256_setzero_pd();

                        r4 = _mm256_setzero_pd();
                        r5 = _mm256_setzero_pd();
                        r6 = _mm256_setzero_pd();
                        r7 = _mm256_setzero_pd();

                        idx_b = j1;
                        for (int k1 = 0; k1 < r; k1++) {
                            a0 = _mm256_set1_pd(W[ri1 + k1]);
                            a1 = _mm256_set1_pd(W[ri1 + r + k1]);

                            b0 = _mm256_load_pd(&denominator_r[idx_b]);
                            b1 = _mm256_load_pd(&denominator_r[idx_b + 4]);
                            b2 = _mm256_load_pd(&denominator_r[idx_b + 8]);
                            b3 = _mm256_load_pd(&denominator_r[idx_b + 12]);

                            r0 = _mm256_fmadd_pd(a0, b0, r0);
                            r1 = _mm256_fmadd_pd(a0, b1, r1);
                            r2 = _mm256_fmadd_pd(a0, b2, r2);
                            r3 = _mm256_fmadd_pd(a0, b3, r3);

                            r4 = _mm256_fmadd_pd(a1, b0, r4);
                            r5 = _mm256_fmadd_pd(a1, b1, r5);
                            r6 = _mm256_fmadd_pd(a1, b2, r6);
                            r7 = _mm256_fmadd_pd(a1, b3, r7);

                            idx_b += r;
                        }

                        _mm256_store_pd(&denominator_W[ri1j1], r0);
                        _mm256_store_pd(&denominator_W[ri1j1 + 4], r1);
                        _mm256_store_pd(&denominator_W[ri1j1 + 8], r2);
                        _mm256_store_pd(&denominator_W[ri1j1 + 12], r3);

                        _mm256_store_pd(&denominator_W[idx_r], r4);
                        _mm256_store_pd(&denominator_W[idx_r + 4], r5);
                        _mm256_store_pd(&denominator_W[idx_r + 8], r6);
                        _mm256_store_pd(&denominator_W[idx_r + 12], r7);
                    }
                    ri1 += r_2;
                }

                //element-wise multiplication and division
                // NEW - vectorized and works in general case (for any m and n)
                ri1 = ri;
                for (int i1 = i; i1 < min(inB, original_m); i1++) {
                    for (int j1 = j; j1 < min(jnB - 3, original_r - (original_r % 4)); j1 += 4) {
                        ri1j1 = ri1 + j1;

                        num = _mm256_loadu_pd(&numerator_W[ri1j1]);
                        fac = _mm256_loadu_pd(&W[ri1j1]);
                        den = _mm256_loadu_pd(&denominator_W[ri1j1]);
                        num = _mm256_mul_pd(fac, num);
                        res = _mm256_div_pd(num, den);
                        _mm256_storeu_pd(&W_new[ri1j1], res);
                    }
                    ri1 += r;
                }
                //Handling the remaining elements
                for (int i1 = i; i1 < original_m; i1++) {
                    for (int j1 = original_r - (original_r % 4); j1 < original_r; j1++) {
                        W_new[i1 * r + j1] = W[i1 * r + j1] * numerator_W[i1 * r + j1] / denominator_W[i1 * r + j1];
                    }
                }


                //computation for Hn+2

                //NEW - Since now only MMmul exists, no rmul, we have to transpose the current block
                //Calculate the transpose of current block of W
                for (int i1 = i; i1 < inB; i1 += 4) {
                    for (int j1 = j; j1 < jnB; j1 += 4) {
                        transpose4x4(&Wt[j1 * m + i1], &W_new[i1 * r + j1], m, r);
                    }
                }

                //WtV mul
                ni1 = nj;
                mi1 = mj;
                for (int i1 = j; i1 <= jnB - 2; i1 += 2) {
                    for (int j1 = 0; j1 <= n - 16; j1 += 16) {
                        ni1j1 = ni1 + j1;
                        idx_r = ni1j1 + n;

                        r0 = _mm256_load_pd(&numerator[ni1j1]);
                        r1 = _mm256_load_pd(&numerator[ni1j1 + 4]);
                        r2 = _mm256_load_pd(&numerator[ni1j1 + 8]);
                        r3 = _mm256_load_pd(&numerator[ni1j1 + 12]);

                        r4 = _mm256_load_pd(&numerator[idx_r]);
                        r5 = _mm256_load_pd(&numerator[idx_r + 4]);
                        r6 = _mm256_load_pd(&numerator[idx_r + 8]);
                        r7 = _mm256_load_pd(&numerator[idx_r + 12]);

                        idx_b = ni + j1;
                        for (int k1 = i; k1 < inB; k1++) {
                            a0 = _mm256_set1_pd(Wt[mi1 + k1]);
                            a1 = _mm256_set1_pd(Wt[mi1 + m + k1]);

                            b0 = _mm256_load_pd(&V[idx_b]);
                            b1 = _mm256_load_pd(&V[idx_b + 4]);
                            b2 = _mm256_load_pd(&V[idx_b + 8]);
                            b3 = _mm256_load_pd(&V[idx_b + 12]);

                            r0 = _mm256_fmadd_pd(a0, b0, r0);
                            r1 = _mm256_fmadd_pd(a0, b1, r1);
                            r2 = _mm256_fmadd_pd(a0, b2, r2);
                            r3 = _mm256_fmadd_pd(a0, b3, r3);

                            r4 = _mm256_fmadd_pd(a1, b0, r4);
                            r5 = _mm256_fmadd_pd(a1, b1, r5);
                            r6 = _mm256_fmadd_pd(a1, b2, r6);
                            r7 = _mm256_fmadd_pd(a1, b3, r7);

                            idx_b += n;
                        }

                        _mm256_store_pd(&numerator[ni1j1], r0);
                        _mm256_store_pd(&numerator[ni1j1 + 4], r1);
                        _mm256_store_pd(&numerator[ni1j1 + 8], r2);
                        _mm256_store_pd(&numerator[ni1j1 + 12], r3);

                        _mm256_store_pd(&numerator[idx_r], r4);
                        _mm256_store_pd(&numerator[idx_r + 4], r5);
                        _mm256_store_pd(&numerator[idx_r + 8], r6);
                        _mm256_store_pd(&numerator[idx_r + 12], r7);
                    }
                    ni1 += n_2;
                    mi1 += m_2;
                }

                //WtW mul
                ri1 = rj;
                mi1 = mj;
                for (int i1 = j; i1 <= jnB - 2; i1 += 2) {
                    for (int j1 = 0; j1 <= jnB - 16; j1 += 16) {
                        ri1j1 = ri1 + j1;
                        idx_r = ri1j1 + r;

                        r0 = _mm256_load_pd(&denominator_l[ri1j1]);
                        r1 = _mm256_load_pd(&denominator_l[ri1j1 + 4]);
                        r2 = _mm256_load_pd(&denominator_l[ri1j1 + 8]);
                        r3 = _mm256_load_pd(&denominator_l[ri1j1 + 12]);

                        r4 = _mm256_load_pd(&denominator_l[idx_r]);
                        r5 = _mm256_load_pd(&denominator_l[idx_r + 4]);
                        r6 = _mm256_load_pd(&denominator_l[idx_r + 8]);
                        r7 = _mm256_load_pd(&denominator_l[idx_r + 12]);

                        idx_b = ri + j1;
                        for (int k1 = i; k1 < inB; k1++) {
                            a0 = _mm256_set1_pd(Wt[mi1 + k1]);
                            a1 = _mm256_set1_pd(Wt[mi1 + m + k1]);

                            b0 = _mm256_load_pd(&W_new[idx_b]);
                            b1 = _mm256_load_pd(&W_new[idx_b + 4]);
                            b2 = _mm256_load_pd(&W_new[idx_b + 8]);
                            b3 = _mm256_load_pd(&W_new[idx_b + 12]);

                            r0 = _mm256_fmadd_pd(a0, b0, r0);
                            r1 = _mm256_fmadd_pd(a0, b1, r1);
                            r2 = _mm256_fmadd_pd(a0, b2, r2);
                            r3 = _mm256_fmadd_pd(a0, b3, r3);

                            r4 = _mm256_fmadd_pd(a1, b0, r4);
                            r5 = _mm256_fmadd_pd(a1, b1, r5);
                            r6 = _mm256_fmadd_pd(a1, b2, r6);
                            r7 = _mm256_fmadd_pd(a1, b3, r7);

                            idx_b += r;
                        }

                        _mm256_store_pd(&denominator_l[ri1j1], r0);
                        _mm256_store_pd(&denominator_l[ri1j1 + 4], r1);
                        _mm256_store_pd(&denominator_l[ri1j1 + 8], r2);
                        _mm256_store_pd(&denominator_l[ri1j1 + 12], r3);

                        _mm256_store_pd(&denominator_l[idx_r], r4);
                        _mm256_store_pd(&denominator_l[idx_r + 4], r5);
                        _mm256_store_pd(&denominator_l[idx_r + 8], r6);
                        _mm256_store_pd(&denominator_l[idx_r + 12], r7);
                    }
                    ri1 += r_2;
                    mi1 += m_2;
                }
                ri1 = mi1 = 0;
                for (int i1 = 0; i1 <= j - 2; i1 += 2) {
                    for (int j1 = j; j1 <= jnB - 16; j1 += 16) {
                        ri1j1 = ri1 + j1;
                        idx_r = ri1j1 + r;

                        r0 = _mm256_load_pd(&denominator_l[ri1j1]);
                        r1 = _mm256_load_pd(&denominator_l[ri1j1 + 4]);
                        r2 = _mm256_load_pd(&denominator_l[ri1j1 + 8]);
                        r3 = _mm256_load_pd(&denominator_l[ri1j1 + 12]);

                        r4 = _mm256_load_pd(&denominator_l[idx_r]);
                        r5 = _mm256_load_pd(&denominator_l[idx_r + 4]);
                        r6 = _mm256_load_pd(&denominator_l[idx_r + 8]);
                        r7 = _mm256_load_pd(&denominator_l[idx_r + 12]);

                        idx_b = ri + j1;
                        for (int k1 = i; k1 < inB; k1++) {
                            a0 = _mm256_set1_pd(Wt[mi1 + k1]);
                            a1 = _mm256_set1_pd(Wt[mi1 + m + k1]);

                            b0 = _mm256_load_pd(&W_new[idx_b]);
                            b1 = _mm256_load_pd(&W_new[idx_b + 4]);
                            b2 = _mm256_load_pd(&W_new[idx_b + 8]);
                            b3 = _mm256_load_pd(&W_new[idx_b + 12]);

                            r0 = _mm256_fmadd_pd(a0, b0, r0);
                            r1 = _mm256_fmadd_pd(a0, b1, r1);
                            r2 = _mm256_fmadd_pd(a0, b2, r2);
                            r3 = _mm256_fmadd_pd(a0, b3, r3);

                            r4 = _mm256_fmadd_pd(a1, b0, r4);
                            r5 = _mm256_fmadd_pd(a1, b1, r5);
                            r6 = _mm256_fmadd_pd(a1, b2, r6);
                            r7 = _mm256_fmadd_pd(a1, b3, r7);

                            idx_b += r;
                        }

                        _mm256_store_pd(&denominator_l[ri1j1], r0);
                        _mm256_store_pd(&denominator_l[ri1j1 + 4], r1);
                        _mm256_store_pd(&denominator_l[ri1j1 + 8], r2);
                        _mm256_store_pd(&denominator_l[ri1j1 + 12], r3);

                        _mm256_store_pd(&denominator_l[idx_r], r4);
                        _mm256_store_pd(&denominator_l[idx_r + 4], r5);
                        _mm256_store_pd(&denominator_l[idx_r + 8], r6);
                        _mm256_store_pd(&denominator_l[idx_r + 12], r7);
                    }
                    ri1 += r_2;
                    mi1 += m_2;
                }
                nj += nnB_j;
                rj += rnB_j;
                mj += mnB_j;
            }
            ri += rnB_i;
            ni += nnB_i;
            mi += mnB_i;
        }

        memcpy(W, W_new, d_mr);
    }

    free(numerator);
    free(denominator);
    free(denominator_l);
    free(denominator_r);
    free(numerator_W);
    free(denominator_W);
    free(Wt);
    free(W_new);
    free(approximation);

    // here i should remove the padding from the matrices
    unpad_matrix(&V, &temp_m, &temp_n, original_m, original_n);
    unpad_matrix(&W, &m, &temp_r, original_m, original_r);
    unpad_matrix(&H, &r, &n, original_r, original_n);

    memcpy(V_final, V, m * n * double_size);
    memcpy(W_final, W, m * r * double_size);
    memcpy(H_final, H, r * n * double_size);

    free(V);
    free(H);
    free(W);

    return err;
}

