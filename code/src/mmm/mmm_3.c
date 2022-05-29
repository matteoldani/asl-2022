#include <mmm/mmm_3.h>
static int double_size = sizeof(double);

// POSSIBLE IMPROVMENT --> ask for alloc_aling 32 bytes, should gain 0.5ish
void matrix_mul_3(double *A, int A_n_row, int A_n_col, double *B, int B_n_row, int B_n_col, double *R, int R_n_row, int R_n_col)
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

