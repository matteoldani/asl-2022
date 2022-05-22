#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

//NEW - optimization done on baseline1

//NEW - all functions moved to the same file and marked as inline

static unsigned int double_size = sizeof(double); //NEW - call sizeof only once and then just use the local variable

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
void matrix_mul_opt0(double *A, int A_n_row, int A_n_col, double*B, int B_n_row, int B_n_col, double*R, int R_n_row, int R_n_col) {
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
void matrix_ltrans_mul_opt0(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col) {

    int Rij;

    for (int i = 0; i < A_n_col; i++) {
        for (int j = 0; j < B_n_col; j++) {
            Rij = i * R_n_col + j;
            R[Rij] = 0;
            for (int k = 0; k < B_n_row; k++)
                R[Rij] += A[k * A_n_col + i] * B[k * B_n_col + j];
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
void matrix_rtrans_mul_opt0(double* A, int A_n_row, int A_n_col, double* B, int B_n_row, int B_n_col, double* R, int R_n_row, int R_n_col) {
    
    int Rij;

    for (int i = 0; i < A_n_row; i++) {
        for (int j = 0; j < B_n_row; j++) {
            Rij = i * R_n_col + j;
            R[Rij] = 0;
            for (int k = 0; k < A_n_col; k++)
                R[Rij] += A[i * A_n_col + k] * B[j * B_n_col + k];
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
inline double error(double* approx, double* V, double* W, double* H, int m, int n, int r, int mn, double norm_V) {

    matrix_mul_opt0(W, m, r, H, r, n, approx, m, n);

    double norm_approx, temp;

    //NEW - removed calls to the norm function, norm now calculated in place
    //NEW - two loops over the approx matrix were replaced with one + scalar replacement added and number of accesses to memory reduced by 1/3 per function call
    norm_approx = 0;
    for (int i = 0; i < mn; i++)
    {
        temp = V[i] - approx[i];
        norm_approx += temp * temp;
    }
    norm_approx = sqrt(norm_approx);

    return norm_approx * norm_V; //NEW - div replaced with mul thanks to precomputing
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
//NEW - removed the usage of structs, a matrix is now just a double*, also m,n,r are just sent once (no more row and col information for each matrix separately)
double nnm_factorization_opt0(double *V, double*W, double*H, int m, int n, int r, int maxIteration, double epsilon) {

    int rn, rr, mr, mn;
    rn = r * n;  //NEW - reduced the number of mul ops by precomputing these values
    rr = r * r;
    mr = m * r;
    mn = m * n;

    //Operands needed to compute Hn+1
    double *numerator, *denominator_l, *denominator;    //r x n, r x r, r x n
    numerator = malloc(double_size * rn);  //NEW - removed calls to allocate functions, allocation is now done directly in place
    denominator_l = malloc(double_size * rr);
    denominator = malloc(double_size * rn);

    //Operands needed to compute Wn+1
    double *numerator_W, *denominator_W, *denominator_l_W;      // m x r, m x r, m x n
    numerator_W = malloc(double_size * mr);
    denominator_W = malloc(double_size * mr);
    denominator_l_W = malloc(double_size * mn);

    double* approximation; //m x n
    approximation = malloc(double_size * mn); //NEW - allocate the approximation matrix only once and send it into the error function

    double norm_V = 0; //NEW - norm for V calculated only once and sent into the error function
    for (int i = 0; i < mn; i++)
        norm_V += V[i] * V[i];
    norm_V = 1 / sqrt(norm_V); //NEW - calculate 1/norm here and avoid having a div instruction inside the error function

    //real convergence computation
    double err = -1;											
    for (int count = 0; count < maxIteration; count++) {
     
        err = error(approximation, V, W, H, m, n, r, mn, norm_V);
        if (err <= epsilon) {
            break;
        }

        //computation for Hn+1
        matrix_ltrans_mul_opt0(W, m, r, V, m, n, numerator, r, n);
        matrix_ltrans_mul_opt0(W, m, r, W, m, r, denominator_l, r, r);
        matrix_mul_opt0(denominator_l, r, r, H, r, n, denominator, r, n);

        //NEW - double loop replaced with a single loop, index calc simplified
        for (int i = 0; i < rn; i++)
            H[i] = H[i] * numerator[i] / denominator[i];

        //computation for Wn+1

        //NEW changed the order of the multiplication to reduce the number of flops
        matrix_rtrans_mul_opt0(V, m, n, H, r, n, numerator_W, m, r);
        matrix_rtrans_mul_opt0(H, r, n, H, r, n, denominator_l, r, r);
        matrix_mul_opt0(W, m, r, denominator_l, r, r, denominator_W, m, r);
        

        for (int i = 0; i < mr; i++)
            W[i] = W[i] * numerator_W[i] / denominator_W[i];
    }

    free(numerator);    //NEW - removed calls to deallocate function, deallocation now done directly in place
    free(denominator);
    free(denominator_l);
    free(numerator_W);
    free(denominator_W);
    free(denominator_l_W);
    
    return err;
}


