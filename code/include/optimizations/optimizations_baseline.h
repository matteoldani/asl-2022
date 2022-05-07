#ifndef ASL_2022_OPTIMIZATIONS_BASELINE_H
#define ASL_2022_OPTIMIZATIONS_BASELINE_H

void matrix_mul(double *A, int A_n_row, int A_n_col, double*B, int B_n_row, int B_n_col, double*R, int R_n_row, int R_n_col);
void matrix_ltrans_mul(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col);
void matrix_rtrans_mul(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col);
double nnm_factorization(double *V, double*W, double*H, int m, int n, int r, int maxIteration, double epsilon);

#endif 