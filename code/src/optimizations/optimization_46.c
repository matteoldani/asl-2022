#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <optimizations/optimizations_46.h>
#include <immintrin.h>

// NEW matrices W, H and V are padded with zeros such that all dimensions are multiple of the blocksize 16. 
//     matrices are unpadded at the end of the nmf computation  


static unsigned int double_size = sizeof(double);


// NEW: the transpose is not optimized because the cleanup loop was not implemented and this version of
//      opt does support any type of inputs
static void transpose(double *src, double *dst,  const int N, const int M) {

    int nB = 1;
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

// NEW: this is the function that pads the matrix to a multiple of the block size
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

    new_Mt = malloc(double_size * temp_c * temp_r);
    transpose(*M, new_Mt, temp_r, *c);
    memset(&new_Mt[temp_r * (*c)], 0, double_size * (temp_c - (*c)) * temp_r);

    free(*M);
    *M = malloc(double_size * temp_c * temp_r);
    *c = temp_c;
    *r = temp_r;
    transpose(new_Mt, *M, temp_c, temp_r); 

    free(new_Mt);


}

// NEW: this is the function that unpard the matrix to the oringinal size
static void unpad_matrix(double **M, int *r, int *c, int original_r, int original_c){

    // lets suppose that are always row majour

    // i can remove the last useless rows
    *M = realloc(*M, (*c) * original_r * double_size);

    // i need to transpose and remove the rest
    double *new_Mt = malloc((*c) * original_r * double_size );
    transpose(*M, new_Mt, original_r, *c);

    // i need to resize the transoposed
    new_Mt = realloc(new_Mt, double_size * original_c * original_r);

    // ie need to transpose back
    free(*M);
    *M = malloc(double_size * original_c * original_r);
    transpose(new_Mt, *M, original_c, original_r);

    *r = original_r;
    *c = original_c;

    free(new_Mt);
    
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
void matrix_mul_opt46_padding(double *A_final, int A_n_row, int A_n_col, double *B_final, int B_n_row, int B_n_col, double *R_final, int R_n_row, int R_n_col)
{   int m = A_n_row;
    int n = B_n_col;
    int r = B_n_row;
    double *V, *H, *W;
    V = malloc(double_size * m * n);
    H = malloc(double_size * r * n);
    W = malloc(double_size * m * r);

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
    
    matrix_mul_opt46(W, m, r, H, r, m, V, m, n);


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
void matrix_mul_opt46(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R, int R_n_row, int R_n_col)
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



    memset(R, 0, double_size * R_n_row * R_n_col);
    //MAIN LOOP BLOCKED 16x16
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

                        _mm256_storeu_pd((double *)&R[Rij], r0);
                        _mm256_storeu_pd((double *)&R[Rij + 4], r1);
                        _mm256_storeu_pd((double *)&R[Rij + 8], r2);
                        _mm256_storeu_pd((double *)&R[Rij + 12], r3);

                        _mm256_storeu_pd((double *)&R[idx_r], r4);
                        _mm256_storeu_pd((double *)&R[idx_r + 4], r5);
                        _mm256_storeu_pd((double *)&R[idx_r + 8], r6);
                        _mm256_storeu_pd((double *)&R[idx_r + 12], r7);

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
inline double error(double* approx, double* V, double* W, double* H, int m, int n, int r, int mn, double norm_V) {

    matrix_mul_opt46(W, m, r, H, r, n, approx, m, n);


    double norm_approx;
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


    int i;
    for (i=0; i<mn; i+=8){
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
double nnm_factorization_opt46(double *V_final, double *W_final, double*H_final, int m, int n, int r, int maxIteration, double epsilon) {

    double *V, *W, *H;

    V = malloc(double_size * m * n);
    H = malloc(double_size * r * n);
    W = malloc(double_size * m * r);

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


    int i;

    ///// NORM

    for (i=0; i<mn; i+=8){
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
        matrix_mul_opt46(Wt, r, m, V, m, n, numerator, r, n);
        matrix_mul_opt46(Wt, r, m, W, m, r, denominator_l, r, r);
        matrix_mul_opt46(denominator_l, r, r, H, r, n, denominator, r, n);
        

        for(i = 0; i < original_r; i ++){
            for(int j = 0; j < original_n; j++){
                H[i * n + j] =   H[i * n + j]   * numerator[i * n + j]   / denominator[i * n + j];
            }
        }

        //computation for Wn+1
        transpose(H, Ht, r, n);
        matrix_mul_opt46(V, m, n, Ht, n, r, numerator_W, m, r);
        matrix_mul_opt46(H, r, n, Ht, n, r, denominator_l, r, r);
        matrix_mul_opt46(W, m, r, denominator_l, r, r, denominator_W, m, r);
      
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
