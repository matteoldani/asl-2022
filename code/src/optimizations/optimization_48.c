#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <optimizations/optimizations_48.h>
#include <immintrin.h>

// NEW WtW and WtV computation is interlieved to avoid reading W twice. 
//     HHt and VHt computation is interlieved to avoid reading V twice.
//     Using vectorized transpose
//     Intermediate results of a blocked MMM stored in buffer 
//     and copied at the end

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

    if((*r % BLOCK_SIZE_MMUL == 0 ) && (*c % BLOCK_SIZE_MMUL == 0 )) {
        return;
    }


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
void matrix_mul_opt48_padding(double *A_final, int A_n_row, int A_n_col, double *B_final, int B_n_row, int B_n_col, double *R_final, int R_n_row, int R_n_col)
{   int m = A_n_row;
    int n = B_n_col;
    int r = B_n_row;
    double *V, *H, *W;
    V = aligned_alloc(32, double_size * m * n);
    H = aligned_alloc(32, double_size * r * n);
    W = aligned_alloc(32, double_size * m * r);

    memcpy(V, R_final, m * n * double_size );
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
    
    matrix_mul_opt48(W, m, r, H, r, m, V, m, n);


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
void matrix_mul_opt48(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R, int R_n_row, int R_n_col)
{  

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
    double buffer[BLOCK_SIZE_MMUL * BLOCK_SIZE_MMUL];

    //MAIN LOOP BLOCKED 16x16
    for (i = 0; i < A_n_row - nB + 1; i += nB)
    {   
        for (j = 0; j < B_n_col - nB + 1; j += nB)
        {   
            memset(buffer, 0, double_size * BLOCK_SIZE_MMUL*BLOCK_SIZE_MMUL);

            for (k = 0; k < A_n_col - nB + 1; k += nB)
            {   

                Rii = 0;
                Aii = Ai;
                for (int ii = i; ii < i + nB - unroll_i + 1; ii += unroll_i)
                {

                    for (int jj = 0; jj < nB - unroll_j + 1; jj += unroll_j)
                    {
                        
                        Rij = Rii + jj;
                        int idx_r = Rij + BLOCK_SIZE_MMUL;
                        
                        r0 = _mm256_loadu_pd((double *)&buffer[Rij]);
                        r1 = _mm256_loadu_pd((double *)&buffer[Rij + 4]);
                        r2 = _mm256_loadu_pd((double *)&buffer[Rij + 8]);
                        r3 = _mm256_loadu_pd((double *)&buffer[Rij + 12]);

                        r4 = _mm256_loadu_pd((double *)&buffer[idx_r]);
                        r5 = _mm256_loadu_pd((double *)&buffer[idx_r + 4]);
                        r6 = _mm256_loadu_pd((double *)&buffer[idx_r + 8]);
                        r7 = _mm256_loadu_pd((double *)&buffer[idx_r + 12]);


                        int idx_b = k*B_n_col + j  + jj;
                        for (kk = k; kk < k + nB; kk++)
                        {
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

                        _mm256_storeu_pd((double *)&buffer[Rij], r0);
                        _mm256_storeu_pd((double *)&buffer[Rij + 4], r1);
                        _mm256_storeu_pd((double *)&buffer[Rij + 8], r2);
                        _mm256_storeu_pd((double *)&buffer[Rij + 12], r3);

                        _mm256_storeu_pd((double *)&buffer[idx_r], r4);
                        _mm256_storeu_pd((double *)&buffer[idx_r + 4], r5);
                        _mm256_storeu_pd((double *)&buffer[idx_r + 8], r6);
                        _mm256_storeu_pd((double *)&buffer[idx_r + 12], r7);

                    }
                    Rii += BLOCK_SIZE_MMUL * unroll_i;
                    Aii += A_n_col * unroll_i;
                }
            }
            Rii = Ri + j;
            for(int is = 0; is < BLOCK_SIZE_MMUL; is++){
                r0 = _mm256_loadu_pd((double *)&buffer[0  + is*BLOCK_SIZE_MMUL]);
                r1 = _mm256_loadu_pd((double *)&buffer[4  + is*BLOCK_SIZE_MMUL]);
                r2 = _mm256_loadu_pd((double *)&buffer[8  + is*BLOCK_SIZE_MMUL]);
                r3 = _mm256_loadu_pd((double *)&buffer[12 + is*BLOCK_SIZE_MMUL]);

                _mm256_storeu_pd((double *)&R[Rii + 0], r0);
                _mm256_storeu_pd((double *)&R[Rii + 4], r1);
                _mm256_storeu_pd((double *)&R[Rii + 8], r2);
                _mm256_storeu_pd((double *)&R[Rii + 12],r3);
                Rii += R_n_col;
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
inline double error(double* approx, double* V, double* W, double* H, int m, int n, int r, int mn, double norm_V) {

    matrix_mul_opt48(W, m, r, H, r, n, approx, m, n);

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
        
        r0 = _mm256_loadu_pd((double *)&V[i]);
        r1 = _mm256_loadu_pd((double *)&V[i + 4]);
        r2 = _mm256_loadu_pd((double *)&V[i + 8]);
        r3 = _mm256_loadu_pd((double *)&V[i + 12]);

        r4 = _mm256_loadu_pd((double *)&approx[i]);
        r5 = _mm256_loadu_pd((double *)&approx[i + 4]);
        r6 = _mm256_loadu_pd((double *)&approx[i + 8]);
        r7 = _mm256_loadu_pd((double *)&approx[i + 12]);

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
double nnm_factorization_opt48(double *V_final, double *W_final, double*H_final, int m, int n, int r, int maxIteration, double epsilon) {

    double *V, *W, *H;
    __m256d zeros = _mm256_setzero_pd();
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

    //Operands needed to compute Hn+1
    double *numerator, *denominator_l, *denominator;    //r x n, r x r, r x n
    numerator = aligned_alloc(32, double_size * rn);
    denominator_l = aligned_alloc(32, double_size * rr);
    denominator = aligned_alloc(32, double_size * rn);

    //Operands needed to compute Wn+1
    double *numerator_W, *denominator_W;      // m x r, m x r, m x n
    numerator_W = aligned_alloc(32, double_size * mr);
    denominator_W = aligned_alloc(32, double_size * mr);

    double* approximation; //m x n
    approximation = aligned_alloc(32, double_size * mn);

    double norm_V  = 0;
    double * norm_tmp = aligned_alloc(32, double_size * 4);
    int i;

    __m256d norm_approx0, norm_approx1, norm_approx2, norm_approx3;
    __m256d t;

    __m256d r0, r1, r2, r3;
    __m256d r4, r5, r6, r7;
    __m256d t0, t1, t2, t3;

    __m256d sum0, sum1, sum2;

    norm_approx0 = _mm256_setzero_pd();
    norm_approx1 = _mm256_setzero_pd();
    norm_approx2 = _mm256_setzero_pd();
    norm_approx3 = _mm256_setzero_pd();

    for (i=0; i<mn; i+=16){
        
        r0 = _mm256_loadu_pd((double *)&V[i]);
        r1 = _mm256_loadu_pd((double *)&V[i + 4]);
        r2 = _mm256_loadu_pd((double *)&V[i + 8]);
        r3 = _mm256_loadu_pd((double *)&V[i + 12]);

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

    //real convergence computation
    double err = -1;	

    double *Wt = aligned_alloc(32, double_size * mr);
    double *Ht = aligned_alloc(32, double_size * rn);		

    for (int count = 0; count < maxIteration; count++) {
        
        err = error(approximation, V, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }

           

        //computation for Hn+1
        
        transpose(W, Wt, m, r);



    int Rij = 0, Ri = 0, Rii,  Ai = 0, Aii;
    int Rij_w = 0, Ri_w = 0, Rii_w;
    int nB = BLOCK_SIZE_MMUL;

    int nBR_n_col = nB * n;
    int nBR_n_col_w = nB * r;
    int nBA_n_col = nB * m;

    int unroll_i = 1, unroll_j = 16;
    int kk, i, j, k;
    

    __m256d a0, a1;
    __m256d b0, b1, b2, b3;

    __m256d r0, r1, r2, r3;
    __m256d r4, r5, r6, r7;

    __m256d r0_w, r1_w, r2_w, r3_w;
    __m256d r4_w, r5_w, r6_w, r7_w;



    //memset(numerator, 0, double_size * rn);
    //memset(denominator_l, 0, double_size * rr);
    //MAIN LOOP BLOCKED 16x16
    double buf_denominator_l[16*16];
    double buf_numerator[16*16];

    for (i = 0; i < r - nB + 1; i += nB)
    {   
        for (j = 0; j < r - nB + 1; j += nB)
        {   
            memset(buf_denominator_l, 0, double_size * BLOCK_SIZE_MMUL*BLOCK_SIZE_MMUL);
            memset(buf_numerator, 0, double_size * BLOCK_SIZE_MMUL *BLOCK_SIZE_MMUL);

            for (k = 0; k < m - nB + 1; k += nB)
            {   
                Rii_w = 0;
                Rii = 0;
                Aii = Ai;
                for (int ii = i; ii < i + nB - unroll_i + 1; ii += unroll_i)
                {

                    for (int jj = 0; jj <  nB - unroll_j + 1; jj += unroll_j)
                    {
                        
                        Rij = Rii + jj;
                        int idx_r = Rij + BLOCK_SIZE_MMUL;

                        Rij_w = Rii_w + jj;
                        int idx_r_w = Rij_w + BLOCK_SIZE_MMUL;
    
                        r0 = _mm256_loadu_pd((double *)&buf_numerator[Rij]);
                        r1 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 4]);
                        r2 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 8]);
                        r3 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 12]);

                        int idx_b = k*n + j + jj;

                        int idx_b_w = k*r + j + jj;
                      
                        r0_w = _mm256_loadu_pd((double *)&buf_denominator_l[Rij_w]);
                        r1_w = _mm256_loadu_pd((double *)&buf_denominator_l[Rij_w + 4]);
                        r2_w = _mm256_loadu_pd((double *)&buf_denominator_l[Rij_w + 8]);
                        r3_w = _mm256_loadu_pd((double *)&buf_denominator_l[Rij_w + 12]);

                       
                        for (kk = k; kk < k + nB; kk++)
                        {   
                            a0 = _mm256_set1_pd(Wt[Aii + kk]);                //Aik0 = A[Aii + kk];
                            
                            b0 = _mm256_loadu_pd((double *)&V[idx_b]);        // Bi0j0 = B[kk * B_n_col + jj];
                            b1 = _mm256_loadu_pd((double *)&V[idx_b + 4]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b2 = _mm256_loadu_pd((double *)&V[idx_b + 8]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b3 = _mm256_loadu_pd((double *)&V[idx_b + 12]);   // Bi0j0 = B[kk * B_n_col + jj];
  
                            r0 = _mm256_fmadd_pd(a0, b0, r0);
                            r1 = _mm256_fmadd_pd(a0, b1, r1);
                            r2 = _mm256_fmadd_pd(a0, b2, r2);
                            r3 = _mm256_fmadd_pd(a0, b3, r3);
      
                            b0 = _mm256_loadu_pd((double *)&W[idx_b_w]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b1 = _mm256_loadu_pd((double *)&W[idx_b_w + 4]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b2 = _mm256_loadu_pd((double *)&W[idx_b_w + 8]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b3 = _mm256_loadu_pd((double *)&W[idx_b_w + 12]);    // Bi0j0 = B[kk * B_n_col + jj];

                            r0_w = _mm256_fmadd_pd(a0, b0, r0_w);
                            r1_w = _mm256_fmadd_pd(a0, b1, r1_w);
                            r2_w = _mm256_fmadd_pd(a0, b2, r2_w);
                            r3_w = _mm256_fmadd_pd(a0, b3, r3_w);
                            idx_b_w += r;

                            idx_b += n;
                        }

       
                        _mm256_storeu_pd((double *)&buf_denominator_l[Rij_w],     r0_w);
                        _mm256_storeu_pd((double *)&buf_denominator_l[Rij_w + 4], r1_w);
                        _mm256_storeu_pd((double *)&buf_denominator_l[Rij_w + 8], r2_w);
                        _mm256_storeu_pd((double *)&buf_denominator_l[Rij_w + 12],r3_w);

                        _mm256_storeu_pd((double *)&buf_numerator[Rij], r0);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 4], r1);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 8], r2);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 12], r3);
                    }
                   
                    Rii_w += BLOCK_SIZE_MMUL * unroll_i;
                    Rii   += BLOCK_SIZE_MMUL * unroll_i;
                    Aii   += m * unroll_i;
                }
                
            }
            Rii_w = Ri_w + j;
            //STORE LOOP
            for(int is = 0; is < BLOCK_SIZE_MMUL; is++){
                r0_w = _mm256_loadu_pd((double *)&buf_denominator_l[0  + is*BLOCK_SIZE_MMUL]);
                r1_w = _mm256_loadu_pd((double *)&buf_denominator_l[4  + is*BLOCK_SIZE_MMUL]);
                r2_w = _mm256_loadu_pd((double *)&buf_denominator_l[8  + is*BLOCK_SIZE_MMUL]);
                r3_w = _mm256_loadu_pd((double *)&buf_denominator_l[12 + is*BLOCK_SIZE_MMUL]);

                _mm256_storeu_pd((double *)&denominator_l[Rii_w + 0], r0_w);
                _mm256_storeu_pd((double *)&denominator_l[Rii_w + 4], r1_w);
                _mm256_storeu_pd((double *)&denominator_l[Rii_w + 8], r2_w);
                _mm256_storeu_pd((double *)&denominator_l[Rii_w + 12],r3_w);
                Rii_w += r;
            }

            Rii = Ri+j;
            for(int is = 0; is < BLOCK_SIZE_MMUL; is++){
                r0 = _mm256_loadu_pd((double *)&buf_numerator[0  + is*BLOCK_SIZE_MMUL]);
                r1 = _mm256_loadu_pd((double *)&buf_numerator[4  + is*BLOCK_SIZE_MMUL]);
                r2 = _mm256_loadu_pd((double *)&buf_numerator[8  + is*BLOCK_SIZE_MMUL]);
                r3 = _mm256_loadu_pd((double *)&buf_numerator[12 + is*BLOCK_SIZE_MMUL]);

                _mm256_storeu_pd((double *)&numerator[Rii + 0], r0);
                _mm256_storeu_pd((double *)&numerator[Rii + 4], r1);
                _mm256_storeu_pd((double *)&numerator[Rii + 8], r2);
                _mm256_storeu_pd((double *)&numerator[Rii + 12],r3);
                Rii += n;
            }
     

        }


        Ri_w += nBR_n_col_w;
        
        Ri += nBR_n_col;
        Ai += nBA_n_col;
    }

    unroll_i = 2;
    Ri = 0;
    Ai = 0;
    for (i = 0; i < r - nB + 1; i += nB)
    {   
        for (; j < n - nB + 1; j += nB)
        {
            memset(buf_numerator, 0, double_size * BLOCK_SIZE_MMUL * BLOCK_SIZE_MMUL);
            for (k = 0; k < m - nB + 1; k += nB)
            {   
                
                Rii = 0;
                Aii = Ai;
                for (int ii = i; ii < i + nB - unroll_i + 1; ii += unroll_i)
                {

                    for (int jj = 0; jj <  nB - unroll_j + 1; jj += unroll_j)
                    {
                        
                        Rij = Rii + jj;
                        int idx_r = Rij + BLOCK_SIZE_MMUL;

                        r0 = _mm256_loadu_pd((double *)&buf_numerator[Rij]);
                        r1 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 4]);
                        r2 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 8]);
                        r3 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 12]);

                        r4 = _mm256_loadu_pd((double *)&buf_numerator[idx_r]);
                        r5 = _mm256_loadu_pd((double *)&buf_numerator[idx_r + 4]);
                        r6 = _mm256_loadu_pd((double *)&buf_numerator[idx_r + 8]);
                        r7 = _mm256_loadu_pd((double *)&buf_numerator[idx_r + 12]);
                        int idx_b = k*n + j + jj;

               
                        for (kk = k; kk < k + nB; kk++)
                        {   
                            a0 = _mm256_set1_pd(Wt[Aii + kk]);                //Aik0 = A[Aii + kk];
                            a1 = _mm256_set1_pd(Wt[Aii + m + kk]);      //Aik1 = A[Aii + A_n_col + kk]; 
                            
                            b0 = _mm256_loadu_pd((double *)&V[idx_b]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b1 = _mm256_loadu_pd((double *)&V[idx_b + 4]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b2 = _mm256_loadu_pd((double *)&V[idx_b + 8]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b3 = _mm256_loadu_pd((double *)&V[idx_b + 12]);    // Bi0j0 = B[kk * B_n_col + jj];
  
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
                        _mm256_storeu_pd((double *)&buf_numerator[Rij], r0);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 4], r1);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 8], r2);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 12], r3);

                        _mm256_storeu_pd((double *)&buf_numerator[idx_r], r4);
                        _mm256_storeu_pd((double *)&buf_numerator[idx_r + 4], r5);
                        _mm256_storeu_pd((double *)&buf_numerator[idx_r + 8], r6);
                        _mm256_storeu_pd((double *)&buf_numerator[idx_r + 12], r7);

                    }
                    Rii   += BLOCK_SIZE_MMUL * unroll_i;
                    Aii   += m * unroll_i;
                }
            }
            //store loop
            Rii = Ri + j;
            for(int is = 0; is < BLOCK_SIZE_MMUL; is++){
                r0 = _mm256_loadu_pd((double *)&buf_numerator[0  + is*BLOCK_SIZE_MMUL]);
                r1 = _mm256_loadu_pd((double *)&buf_numerator[4  + is*BLOCK_SIZE_MMUL]);
                r2 = _mm256_loadu_pd((double *)&buf_numerator[8  + is*BLOCK_SIZE_MMUL]);
                r3 = _mm256_loadu_pd((double *)&buf_numerator[12 + is*BLOCK_SIZE_MMUL]);

                _mm256_storeu_pd((double *)&numerator[Rii + 0], r0);
                _mm256_storeu_pd((double *)&numerator[Rii + 4], r1);
                _mm256_storeu_pd((double *)&numerator[Rii + 8], r2);
                _mm256_storeu_pd((double *)&numerator[Rii + 12],r3);
                Rii += n;
            }
        }
        
        Ri += nBR_n_col;
        Ai += nBA_n_col;
    }
    


    //TODO interlieve this computation
    matrix_mul_opt48(denominator_l, r, r, H, r, n, denominator, r, n);
    

    for(i = 0; i < original_r; i ++){
        for(int j = 0; j < original_n; j++){
            H[i * n + j] =   H[i * n + j]   * numerator[i * n + j]   / denominator[i * n + j];
        }
    }

        //computation for Wn+1
    transpose(H, Ht, r, n);

    // matrix_mul_opt48(V, m, n, Ht, n, r, numerator_W, m, r);
    // matrix_mul_opt48(H, r, n, Ht, n, r, denominator_l, r, r);

    nBR_n_col_w = r * nB;
    
    nBR_n_col   = r * nB;
    nBA_n_col   = n * nB;


    Ai = 0; 
    Ri = 0;
    Ri_w = 0;
    unroll_i = 1;
    for (i = 0; i < r - nB + 1; i += nB)
    {   
        for (j = 0; j < r - nB + 1; j += nB)
        {   
            memset(buf_denominator_l, 0, double_size * BLOCK_SIZE_MMUL*BLOCK_SIZE_MMUL);
            memset(buf_numerator, 0, double_size * BLOCK_SIZE_MMUL *BLOCK_SIZE_MMUL);

            for (k = 0; k < n - nB + 1; k += nB)
            {   
                Rii_w = 0;
                Rii = 0;
                Aii = Ai;
                //Aii_h = Ai_h;
                for (int ii = i; ii < i + nB - unroll_i + 1; ii += unroll_i)
                {

                    for (int jj = 0; jj <  nB - unroll_j + 1; jj += unroll_j)
                    {
                        
                        Rij = Rii + jj;
                        int idx_r = Rij + BLOCK_SIZE_MMUL;

                        Rij_w = Rii_w + jj;
                        int idx_r_w = Rij_w + BLOCK_SIZE_MMUL;
    
                        r0 = _mm256_loadu_pd((double *)&buf_numerator[Rij]);
                        r1 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 4]);
                        r2 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 8]);
                        r3 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 12]);

                        int idx_b = k*r + j + jj;

                        int idx_b_w = k*r + j + jj;
                      
                        r0_w = _mm256_loadu_pd((double *)&buf_denominator_l[Rij_w]);
                        r1_w = _mm256_loadu_pd((double *)&buf_denominator_l[Rij_w + 4]);
                        r2_w = _mm256_loadu_pd((double *)&buf_denominator_l[Rij_w + 8]);
                        r3_w = _mm256_loadu_pd((double *)&buf_denominator_l[Rij_w + 12]);

                       
                        for (kk = k; kk < k + nB; kk++)
                        {   
                            a0 = _mm256_set1_pd(V[Aii + kk]);                //Aik0 = A[Aii + kk];
                            b0 = _mm256_loadu_pd((double *)&Ht[idx_b]);        // Bi0j0 = B[kk * B_n_col + jj];
                            b1 = _mm256_loadu_pd((double *)&Ht[idx_b + 4]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b2 = _mm256_loadu_pd((double *)&Ht[idx_b + 8]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b3 = _mm256_loadu_pd((double *)&Ht[idx_b + 12]);   // Bi0j0 = B[kk * B_n_col + jj];
  
                            r0 = _mm256_fmadd_pd(a0, b0, r0);
                            r1 = _mm256_fmadd_pd(a0, b1, r1);
                            r2 = _mm256_fmadd_pd(a0, b2, r2);
                            r3 = _mm256_fmadd_pd(a0, b3, r3);
      
                            a0 = _mm256_set1_pd(H[Aii + kk]);   
                          

                            r0_w = _mm256_fmadd_pd(a0, b0, r0_w);
                            r1_w = _mm256_fmadd_pd(a0, b1, r1_w);
                            r2_w = _mm256_fmadd_pd(a0, b2, r2_w);
                            r3_w = _mm256_fmadd_pd(a0, b3, r3_w);
                            idx_b_w += r;

                            idx_b += r;
                        }

       
                        _mm256_storeu_pd((double *)&buf_denominator_l[Rij_w],     r0_w);
                        _mm256_storeu_pd((double *)&buf_denominator_l[Rij_w + 4], r1_w);
                        _mm256_storeu_pd((double *)&buf_denominator_l[Rij_w + 8], r2_w);
                        _mm256_storeu_pd((double *)&buf_denominator_l[Rij_w + 12],r3_w);

                        _mm256_storeu_pd((double *)&buf_numerator[Rij], r0);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 4], r1);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 8], r2);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 12], r3);
                    }
                   
                    Rii_w += BLOCK_SIZE_MMUL * unroll_i;
                    Rii   += BLOCK_SIZE_MMUL * unroll_i;
                    Aii   += n * unroll_i;
                 
                }
                
            }
            Rii_w = Ri_w + j;
            //STORE LOOP
            for(int is = 0; is < BLOCK_SIZE_MMUL; is++){
                r0_w = _mm256_loadu_pd((double *)&buf_denominator_l[0  + is*BLOCK_SIZE_MMUL]);
                r1_w = _mm256_loadu_pd((double *)&buf_denominator_l[4  + is*BLOCK_SIZE_MMUL]);
                r2_w = _mm256_loadu_pd((double *)&buf_denominator_l[8  + is*BLOCK_SIZE_MMUL]);
                r3_w = _mm256_loadu_pd((double *)&buf_denominator_l[12 + is*BLOCK_SIZE_MMUL]);

                _mm256_storeu_pd((double *)&denominator_l[Rii_w + 0], r0_w);
                _mm256_storeu_pd((double *)&denominator_l[Rii_w + 4], r1_w);
                _mm256_storeu_pd((double *)&denominator_l[Rii_w + 8], r2_w);
                _mm256_storeu_pd((double *)&denominator_l[Rii_w + 12],r3_w);
                Rii_w += r;
            }

            Rii = Ri+j;
            for(int is = 0; is < BLOCK_SIZE_MMUL; is++){
                r0 = _mm256_loadu_pd((double *)&buf_numerator[0  + is*BLOCK_SIZE_MMUL]);
                r1 = _mm256_loadu_pd((double *)&buf_numerator[4  + is*BLOCK_SIZE_MMUL]);
                r2 = _mm256_loadu_pd((double *)&buf_numerator[8  + is*BLOCK_SIZE_MMUL]);
                r3 = _mm256_loadu_pd((double *)&buf_numerator[12 + is*BLOCK_SIZE_MMUL]);

                _mm256_storeu_pd((double *)&numerator_W[Rii + 0], r0);
                _mm256_storeu_pd((double *)&numerator_W[Rii + 4], r1);
                _mm256_storeu_pd((double *)&numerator_W[Rii + 8], r2);
                _mm256_storeu_pd((double *)&numerator_W[Rii + 12],r3);
                Rii += r;
            }
     

        }


        Ri_w += nBR_n_col_w;
        
        Ri += nBR_n_col;
        Ai += nBA_n_col;
    }
  
    unroll_i = 2;
   
    for (; i < m - nB + 1; i += nB)
    {   
        for (j = 0; j < r - nB + 1; j += nB)
        {
            memset(buf_numerator, 0, double_size * BLOCK_SIZE_MMUL * BLOCK_SIZE_MMUL);
            for (k = 0; k < n - nB + 1; k += nB)
            {   
                
                Rii = 0;
                Aii = Ai;
                for (int ii = i; ii < i + nB - unroll_i + 1; ii += unroll_i)
                {

                    for (int jj = 0; jj <  nB - unroll_j + 1; jj += unroll_j)
                    {
                        
                        Rij = Rii + jj;
                        int idx_r = Rij + BLOCK_SIZE_MMUL;

                        r0 = _mm256_loadu_pd((double *)&buf_numerator[Rij]);
                        r1 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 4]);
                        r2 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 8]);
                        r3 = _mm256_loadu_pd((double *)&buf_numerator[Rij + 12]);

                        r4 = _mm256_loadu_pd((double *)&buf_numerator[idx_r]);
                        r5 = _mm256_loadu_pd((double *)&buf_numerator[idx_r + 4]);
                        r6 = _mm256_loadu_pd((double *)&buf_numerator[idx_r + 8]);
                        r7 = _mm256_loadu_pd((double *)&buf_numerator[idx_r + 12]);
                        int idx_b = k*r + j + jj;

               
                        for (kk = k; kk < k + nB; kk++)
                        {   
                            a0 = _mm256_set1_pd(V[Aii + kk]);                //Aik0 = A[Aii + kk];
                            a1 = _mm256_set1_pd(V[Aii + n + kk]);      //Aik1 = A[Aii + A_n_col + kk]; 
                            
                            b0 = _mm256_loadu_pd((double *)&Ht[idx_b]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b1 = _mm256_loadu_pd((double *)&Ht[idx_b + 4]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b2 = _mm256_loadu_pd((double *)&Ht[idx_b + 8]);    // Bi0j0 = B[kk * B_n_col + jj];
                            b3 = _mm256_loadu_pd((double *)&Ht[idx_b + 12]);    // Bi0j0 = B[kk * B_n_col + jj];
  
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
                        _mm256_storeu_pd((double *)&buf_numerator[Rij], r0);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 4], r1);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 8], r2);
                        _mm256_storeu_pd((double *)&buf_numerator[Rij + 12], r3);

                        _mm256_storeu_pd((double *)&buf_numerator[idx_r], r4);
                        _mm256_storeu_pd((double *)&buf_numerator[idx_r + 4], r5);
                        _mm256_storeu_pd((double *)&buf_numerator[idx_r + 8], r6);
                        _mm256_storeu_pd((double *)&buf_numerator[idx_r + 12], r7);

                    }
                    Rii   += BLOCK_SIZE_MMUL * unroll_i;
                    Aii   += n * unroll_i;
                }
            }
            //store loop
            Rii = Ri + j;
            for(int is = 0; is < BLOCK_SIZE_MMUL; is++){
                r0 = _mm256_loadu_pd((double *)&buf_numerator[0  + is*BLOCK_SIZE_MMUL]);
                r1 = _mm256_loadu_pd((double *)&buf_numerator[4  + is*BLOCK_SIZE_MMUL]);
                r2 = _mm256_loadu_pd((double *)&buf_numerator[8  + is*BLOCK_SIZE_MMUL]);
                r3 = _mm256_loadu_pd((double *)&buf_numerator[12 + is*BLOCK_SIZE_MMUL]);

                _mm256_storeu_pd((double *)&numerator_W[Rii + 0], r0);
                _mm256_storeu_pd((double *)&numerator_W[Rii + 4], r1);
                _mm256_storeu_pd((double *)&numerator_W[Rii + 8], r2);
                _mm256_storeu_pd((double *)&numerator_W[Rii + 12],r3);
                Rii += r;
            }
        }
        
        Ri += nBR_n_col;
        Ai += nBA_n_col;
    }
    






        matrix_mul_opt48(W, m, r, denominator_l, r, r, denominator_W, m, r);
      
        for(i = 0; i < original_m; i ++){
            for(int j = 0; j < original_r; j++){
                W[i * r + j] =   W[i * r + j]   * numerator_W[i * r + j]   / denominator_W[i * r + j];
            }
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
