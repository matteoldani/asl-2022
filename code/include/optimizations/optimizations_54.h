#include <asl.h>

void matrix_mul_opt54(double *A, int A_n_row, int A_n_col, double*B, int B_n_row, int B_n_col, double*R, int R_n_row, int R_n_col);
double nnm_factorization_opt54(double *V, double*W, double*H, int m, int n, int r, int maxIteration, double epsilon);
void matrix_mul_opt54_padding(double *A_final, int A_n_row, int A_n_col, double *B_final, int B_n_row, int B_n_col,
                              double *R_final, int R_n_row, int R_n_col);
