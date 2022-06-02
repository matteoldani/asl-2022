#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <optimizations/optimizations_54.h>
#include <immintrin.h>

//NEW - On top of opt_51 generalised so that it works for arbitrary rank input

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

static void pad_matrix(double ** M, int *r, int *c){
    int temp_r;
    int temp_c;

    if( (*r) %BLOCK_SIZE_MMUL != 0){
        temp_r = (((*r) / BLOCK_SIZE_MMUL ) + 1)*BLOCK_SIZE_MMUL;   
    }else{
        temp_r = *r;
    }

    if((*c)%BLOCK_SIZE_MMUL != 0){
        temp_c = (((*c )/ BLOCK_SIZE_MMUL) + 1) * BLOCK_SIZE_MMUL;
    }else{
        temp_c = *c;
    }

    double *new_Mt;

    *M = realloc(*M, double_size * (*c) * temp_r);
    // i need to pad the rows before and the cols after transposing
    memset(&(*M)[(*c)*(*r)], 0, double_size * (temp_r-(*r)) * (*c));

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

static void unpad_matrix(double **M, int *r, int *c, int original_r, int original_c){

    // lets suppose that are always row majour

    // i can remove the last useless rows
    *M = realloc(*M, (*c) * original_r * double_size);

    // i need to transpose and remove the rest
    double *new_Mt = aligned_alloc(32, (*c) * original_r * double_size );
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
inline void matrix_mul_opt54(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R, int R_n_row, int R_n_col) {  

    //NEW to this matrix mult will arrive only padded matrices thus we don't need the cleanups loops
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
                        
                        r0 = _mm256_load_pd((double *)&R[Rij]);
                        r1 = _mm256_load_pd((double *)&R[Rij + 4]);
                        r2 = _mm256_load_pd((double *)&R[Rij + 8]);
                        r3 = _mm256_load_pd((double *)&R[Rij + 12]);

                        r4 = _mm256_load_pd((double *)&R[idx_r]);
                        r5 = _mm256_load_pd((double *)&R[idx_r + 4]);
                        r6 = _mm256_load_pd((double *)&R[idx_r + 8]);
                        r7 = _mm256_load_pd((double *)&R[idx_r + 12]);

                        int idx_b = k*B_n_col + jj;
                        for (kk = k; kk < k + nB; kk++) {
                            a0 = _mm256_set1_pd(A[Aii + kk]);                //Aik0 = A[Aii + kk];
                            a1 = _mm256_set1_pd(A[Aii + A_n_col + kk]);      //Aik1 = A[Aii + A_n_col + kk]; 
                            
                            b0 = _mm256_load_pd((double *)&B[idx_b]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b1 = _mm256_load_pd((double *)&B[idx_b + 4]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b2 = _mm256_load_pd((double *)&B[idx_b + 8]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b3 = _mm256_load_pd((double *)&B[idx_b + 12]);    // Bi0j0 = B[kk * B_n_col + jj];
  
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

                        _mm256_store_pd((double *)&R[Rij], r0);
                        _mm256_store_pd((double *)&R[Rij + 4], r1);
                        _mm256_store_pd((double *)&R[Rij + 8], r2);
                        _mm256_store_pd((double *)&R[Rij + 12], r3);

                        _mm256_store_pd((double *)&R[idx_r], r4);
                        _mm256_store_pd((double *)&R[idx_r + 4], r5);
                        _mm256_store_pd((double *)&R[idx_r + 8], r6);
                        _mm256_store_pd((double *)&R[idx_r + 12], r7);
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
    matrix_mul_opt54(W, m, r, H, r, n, approx, m, n);

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
    for (i=0; i<mn; i+=16){
        
        r0 = _mm256_load_pd((double *)&V[i]);
        r1 = _mm256_load_pd((double *)&V[i + 4]);
        r2 = _mm256_load_pd((double *)&V[i + 8]);
        r3 = _mm256_load_pd((double *)&V[i + 12]);

        r4 = _mm256_load_pd((double *)&approx[i]);
        r5 = _mm256_load_pd((double *)&approx[i + 4]);
        r6 = _mm256_load_pd((double *)&approx[i + 8]);
        r7 = _mm256_load_pd((double *)&approx[i + 12]);

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

    _mm256_store_pd(&norm[0], t);
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
double nnm_factorization_opt54(double *V_final, double *W_final, double*H_final, int m, int n, int r, int maxIteration, double epsilon) {

    double *V, *W, *H;

    V = aligned_alloc(32, double_size * m * n);
    H = aligned_alloc(32, double_size * r * n);
    W = aligned_alloc(32, double_size * m * r);

    memcpy(V, V_final, m * n * double_size );
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
    double *numerator, *denominator_l, *denominator;    //r x n, r x r, r x n
    numerator = aligned_alloc(32, d_rn);
    denominator_l = aligned_alloc(32, d_rr);
    denominator = aligned_alloc(32, d_rn);

    //Operands needed to compute Wn+1
    double *numerator_W, *denominator_W, *denominator_r;      // m x r, m x r, r x r
    numerator_W = aligned_alloc(32, d_mr);
    denominator_W = aligned_alloc(32, d_mr);
    denominator_r = aligned_alloc(32, d_rr);

    double* approximation; //m x n
    approximation = aligned_alloc(32, d_mn);

    double norm_V  = 0;
    double * norm_tmp = aligned_alloc(32, double_size * 4);
    int i;

    __m256d norm_approx0, norm_approx1, norm_approx2, norm_approx3;
    __m256d t;

    __m256d r0, r1, r2, r3;

    __m256d sum0, sum1;

    norm_approx0 = _mm256_setzero_pd();
    norm_approx1 = _mm256_setzero_pd();
    norm_approx2 = _mm256_setzero_pd();
    norm_approx3 = _mm256_setzero_pd();

    for (i=0; i<mn; i+=16){
        
        r0 = _mm256_load_pd((double *)&V[i]);
        r1 = _mm256_load_pd((double *)&V[i + 4]);
        r2 = _mm256_load_pd((double *)&V[i + 8]);
        r3 = _mm256_load_pd((double *)&V[i + 12]);

        norm_approx0 = _mm256_fmadd_pd(r0, r0, norm_approx0);
        norm_approx1 = _mm256_fmadd_pd(r1, r1, norm_approx1);
        norm_approx2 = _mm256_fmadd_pd(r2, r2, norm_approx2);
        norm_approx3 = _mm256_fmadd_pd(r3, r3, norm_approx3);
    }


    sum0 = _mm256_add_pd(norm_approx0, norm_approx1);
    sum1 = _mm256_add_pd(norm_approx2, norm_approx3);
    sum0 = _mm256_add_pd(sum0, sum1);

    t = _mm256_hadd_pd(sum0, sum0);
    _mm256_store_pd(&norm_tmp[0], t);
    norm_V = 1 / sqrt(norm_tmp[0] + norm_tmp[2]);

    //real convergence computation
    double err = -1;	

    double *Wt = aligned_alloc(32, d_mr);
    double *Ht = aligned_alloc(32, d_rn);
    double *H_new = aligned_alloc(32, d_rn);

    int nB_i = BLOCK_SIZE_H_ROW;
    int nB_j = BLOCK_SIZE_H_COL;
    int inB, jnB, mnB_i = m * nB_i, rnB_i = r * nB_i, nnB_i = n * nB_i, n_2 = n << 1, r_2 = r << 1;
    int ri, mi, ni, ri1, ni1, ni1j1, ri1j1, idx_r, idx_b;
    int limit_orignial_n = original_n - (original_n % 4);
    int min_i, min_j;

    __m256d num_1, fac_1, den_1, res_1;
    __m256d a0, a1;
    __m256d b0, b1, b2, b3;
    __m256d r4, r5, r6, r7;

    for (int count = 0; count < maxIteration; count++) {
        
        err = error(approximation, V, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }

        memset(numerator_W, 0, d_mr);
        memset(denominator_r, 0, d_rr);

        transpose(W, Wt, m, r);

        //computation for Hn+1
        matrix_mul_opt54(Wt, r, m, W, m, r, denominator_l, r, r);
        matrix_mul_opt54(Wt, r, m, V, m, n, numerator, r, n);

        //NEW - All operations done on blocks are now done optimally - using vector instructions
        ri = mi = ni = 0;
        for (int i = 0; i < r; i += nB_i) {
            inB = i + nB_i;
            min_i = min(inB, original_r);
            for (int j = 0; j < n; j += nB_j) {
                jnB = j + nB_j;
                min_j = min(jnB - 3, limit_orignial_n);

                //computation for Hn+1
                //(WtW)*H mul and element-wise multiplication and division
                ni1 = ni;
                ri1 = ri;
                for (int i1 = i; i1 <= inB - 2; i1 += 2) {
                    for (int j1 = j; j1 <= jnB - 16; j1 += 16) {
                        ni1j1 = ni1 + j1;
                        idx_r = ni1j1 + n;

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
                            a0 = _mm256_set1_pd(denominator_l[ri1 + k1]);
                            a1 = _mm256_set1_pd(denominator_l[ri1 + r + k1]);

                            b0 = _mm256_load_pd(&H[idx_b]);
                            b1 = _mm256_load_pd(&H[idx_b + 4]);
                            b2 = _mm256_load_pd(&H[idx_b + 8]);
                            b3 = _mm256_load_pd(&H[idx_b + 12]);

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
                        _mm256_store_pd(&denominator[ni1j1], r0);
                        _mm256_store_pd(&denominator[ni1j1 + 4], r1);
                        _mm256_store_pd(&denominator[ni1j1 + 8], r2);
                        _mm256_store_pd(&denominator[ni1j1 + 12], r3);

                        _mm256_store_pd(&denominator[idx_r], r4);
                        _mm256_store_pd(&denominator[idx_r + 4], r5);
                        _mm256_store_pd(&denominator[idx_r + 8], r6);
                        _mm256_store_pd(&denominator[idx_r + 12], r7);
                    }
                    ni1 += n_2;
                    ri1 += r_2;
                }

                //element-wise multiplication and division
                //NEW Fixed for arbitrary rank input
                ni1 = ni;
                for (int i1 = i; i1 < min_i; i1++) {
                    for (int j1 = j; j1 < min_j; j1+=4) {
                        ni1j1 = ni1 + j1;
                        
                        num_1 = _mm256_load_pd(&numerator[ni1j1]);
                        fac_1 = _mm256_load_pd(&H[ni1j1]);
                        den_1 = _mm256_load_pd(&denominator[ni1j1]);
                        num_1 = _mm256_mul_pd(fac_1, num_1);
                        res_1 = _mm256_div_pd(num_1, den_1);
                        _mm256_store_pd(&H_new[ni1j1], res_1);
                    }
                    ni1 += n;
                }

                ni1 = ni;
                for (int i1 = i; i1 < original_r; i1++) {
                    for (int j1 = limit_orignial_n; j1 < original_n; j1++) {
                        ni1j1 = ni1 + j1;
                        H_new[ni1j1] = H[ni1j1] * numerator[ni1j1] / denominator[ni1j1];
                    }
                    ni1 += n;
                }
                
                //Calculate the transpose of current block of H
                for (int i1 = i; i1 < inB; i1 += 4) {
                    for (int j1 = j; j1 < jnB; j1 += 4)
                        transpose4x4(&Ht[j1 * r + i1], &H_new[i1 * n + j1], r, n);
                }


                //computation for Wn+1

                //V*Ht mul
                ri1 = ni1 = 0;
                for (int i1 = 0; i1 <= m - 2; i1 += 2) {
                    for (int j1 = i; j1 <= inB - 16; j1 += 16) {
                        ri1j1 = ri1 + j1;
                        idx_r = ri1j1 + r;

                        r0 = _mm256_load_pd(&numerator_W[ri1j1]);
                        r1 = _mm256_load_pd(&numerator_W[ri1j1 + 4]);
                        r2 = _mm256_load_pd(&numerator_W[ri1j1 + 8]);
                        r3 = _mm256_load_pd(&numerator_W[ri1j1 + 12]);

                        r4 = _mm256_load_pd(&numerator_W[idx_r]);
                        r5 = _mm256_load_pd(&numerator_W[idx_r + 4]);
                        r6 = _mm256_load_pd(&numerator_W[idx_r + 8]);
                        r7 = _mm256_load_pd(&numerator_W[idx_r + 12]);

                        idx_b = j * r + j1;
                        for (int k1 = j; k1 < jnB; k1++) {
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
                    ri1 += r_2;
                    ni1 += n_2;
                }

                //H*Ht mul
                ni1 = ri1 = 0;
                for (int i1 = 0; i1 <= inB - 2; i1 += 2) {
                    for (int j1 = i; j1 <= inB - 16; j1 += 16) {
                        ri1j1 = ri1 + j1;
                        idx_r = ri1j1 + r;

                        r0 = _mm256_load_pd(&denominator_r[ri1j1]);
                        r1 = _mm256_load_pd(&denominator_r[ri1j1 + 4]);
                        r2 = _mm256_load_pd(&denominator_r[ri1j1 + 8]);
                        r3 = _mm256_load_pd(&denominator_r[ri1j1 + 12]);

                        r4 = _mm256_load_pd(&denominator_r[idx_r]);
                        r5 = _mm256_load_pd(&denominator_r[idx_r + 4]);
                        r6 = _mm256_load_pd(&denominator_r[idx_r + 8]);
                        r7 = _mm256_load_pd(&denominator_r[idx_r + 12]);

                        idx_b = j * r + j1;
                        for (int k1 = j; k1 < jnB; k1++) {
                            a0 = _mm256_set1_pd(H_new[ni1 + k1]);
                            a1 = _mm256_set1_pd(H_new[ni1 + n + k1]);

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

                        _mm256_store_pd(&denominator_r[ri1j1], r0);
                        _mm256_store_pd(&denominator_r[ri1j1 + 4], r1);
                        _mm256_store_pd(&denominator_r[ri1j1 + 8], r2);
                        _mm256_store_pd(&denominator_r[ri1j1 + 12], r3);

                        _mm256_store_pd(&denominator_r[idx_r], r4);
                        _mm256_store_pd(&denominator_r[idx_r + 4], r5);
                        _mm256_store_pd(&denominator_r[idx_r + 8], r6);
                        _mm256_store_pd(&denominator_r[idx_r + 12], r7);
                    }
                    ni1 += n_2;
                    ri1 += r_2;
                }
                ni1 = ni;
                ri1 = ri;
                for (int i1 = i; i1 <= inB - 2; i1 += 2) {
                    for (int j1 = 0; j1 <= i - 16; j1 += 16) {
                        ri1j1 = ri1 + j1;
                        idx_r = ri1j1 + r;

                        r0 = _mm256_load_pd(&denominator_r[ri1j1]);
                        r1 = _mm256_load_pd(&denominator_r[ri1j1 + 4]);
                        r2 = _mm256_load_pd(&denominator_r[ri1j1 + 8]);
                        r3 = _mm256_load_pd(&denominator_r[ri1j1 + 12]);

                        r4 = _mm256_load_pd(&denominator_r[idx_r]);
                        r5 = _mm256_load_pd(&denominator_r[idx_r + 4]);
                        r6 = _mm256_load_pd(&denominator_r[idx_r + 8]);
                        r7 = _mm256_load_pd(&denominator_r[idx_r + 12]);

                        idx_b = j * r + j1;
                        for (int k1 = j; k1 < jnB; k1++) {
                            a0 = _mm256_set1_pd(H_new[ni1 + k1]);
                            a1 = _mm256_set1_pd(H_new[ni1 + n + k1]);

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
                        
                        _mm256_store_pd(&denominator_r[ri1j1], r0);
                        _mm256_store_pd(&denominator_r[ri1j1 + 4], r1);
                        _mm256_store_pd(&denominator_r[ri1j1 + 8], r2);
                        _mm256_store_pd(&denominator_r[ri1j1 + 12], r3);

                        _mm256_store_pd(&denominator_r[idx_r], r4);
                        _mm256_store_pd(&denominator_r[idx_r + 4], r5);
                        _mm256_store_pd(&denominator_r[idx_r + 8], r6);
                        _mm256_store_pd(&denominator_r[idx_r + 12], r7);
                    }
                    ri1 += r_2;
                    ni1 += n_2;
                }
            }
            ri += rnB_i;
            mi += mnB_i;
            ni += nnB_i;
        }

        //remaining computation for Wn+1
        matrix_mul_opt54(W, m, r, denominator_r, r, r, denominator_W, m, r);

        for(i = 0; i < original_m; i ++){
            for(int j = 0; j < original_r; j++){
                W[i * r + j] = W[i * r + j] * numerator_W[i * r + j] / denominator_W[i * r + j];
            }
        }
        
        memcpy(H, H_new, d_rn);
    }

    free(numerator);
    free(denominator);
    free(denominator_l);
    free(numerator_W);
    free(denominator_W);
    free(denominator_r);
    free(approximation);
    free(Wt);
    free(Ht);
    free(H_new);

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
