#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "baseline2.h"
#include <string.h>
#if __APPLE__
#include <Accelerate/Accelerate.h>
#else

#include "cblas.h"

#endif

void v_matrix_mul(vMatrix *A, vMatrix *B, vMatrix *R);

void v_matrix_ltrans_mul(vMatrix *A, vMatrix *B, vMatrix *R);

void v_matrix_rtrans_mul(vMatrix *A, vMatrix *B, vMatrix *R);

void print_v_matrix(vMatrix *matrix);

double v_error(vMatrix *V, vMatrix *W, vMatrix *H);

double v_norm(vMatrix *matrix);

// _____________________________ MATRIX MUL _____________________________
/**
 * @brief compute the multiplication of A and B
 * @param A is the first factor
 * @param B is the other factor of the multiplication
 * @param R is the matrix that will hold the result
 */

// RowMajor implementation
void matrix_mul_rm(vMatrix *A, vMatrix *B, vMatrix *R) {

    // cost: B_col*A_col + 2*A_row*A_col*B_col
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A->n_row, B->n_col, A->n_col, 1,
                A->M, A->n_col, B->M, B->n_col,
                0, R->M, B->n_col);
}

// Straightforward implementation (no BLAS)
void matrix_mul_s(vMatrix *A, vMatrix *B, vMatrix *R) {

    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i * R->n_col + j] += A->M[i * A->n_col + k] * B->M[k * B->n_col + j];

            }

        }
    }
}

// Working impl
void v_matrix_mul(vMatrix *A, vMatrix *B, vMatrix *R) {
     // cost: B_col*A_col + 2*A_row*A_col*B_col
    matrix_mul_rm(A, B, R);
}
//_____________________________________________________________________________


// _____________________________ LEFT MATRIX MUL _____________________________
/**
 * @brief compute the multiplication of A and B transposed
 * @param A is the other factor of the multiplication
 * @param B is the matrix to be transposed
 * @param R is the matrix that will hold the result
 */

// ColMajor impl  ----Param num 10 has an illegal value (0.270872)
// m=n=100        ----Param num 10 has an illegal value (nan)
void matrix_ltrans_mul_cm(vMatrix *A, vMatrix *B, vMatrix *R) {

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                A->n_col, B->n_col, A->n_row, 1,
                A->M, A->n_col, B->M, B->n_row,
                0, R->M, A->n_col);
}

// RowMajor impl (0.47)
void matrix_ltrans_mul_rm(vMatrix *A, vMatrix *B, vMatrix *R) {
     // cost: B_col*B_row + 2*B_row*A_col*B_col
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                A->n_col, B->n_col, B->n_row, 1, //r=A->n_row = B->n_row
                A->M, A->n_col, B->M, B->n_col,
                0, R->M, B->n_col);
}

// Straightforward implementation (no BLAS)
void matrix_ltrans_mul_s(vMatrix *A, vMatrix *B, vMatrix *R) {

    for (int i = 0; i < A->n_col; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < B->n_row; k++) {
                R->M[i * R->n_col + j] += A->M[k * A->n_col + i] * B->M[k * B->n_col + j];
            }
        }
    }
}

// Working impl
void v_matrix_ltrans_mul(vMatrix *A, vMatrix *B, vMatrix *R) {
   
    matrix_ltrans_mul_rm(A, B, R);
}
//_____________________________________________________________________________


// _____________________________ RIGHT MATRIX MUL _____________________________
/**
 * @brief compute the multiplication of A transposed and B
 * @param A is the matrix to be transposed
 * @param B is the other factor of the multiplication
 * @param R is the matrix that will hold the result
 */

// ColMajor impl  ----Param num 8 has an illegal value (0.327659)
void matrix_rtrans_mul_cm(vMatrix *A, vMatrix *B, vMatrix *R) {

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                A->n_row, B->n_row, A->n_col, 1,
                A->M, A->n_row, B->M, B->n_col,
                0, R->M, A->n_row);
}

// RowMajor impl  ----Param num 10 has an illegal value (0.475054)
void matrix_rtrans_mul_rm(vMatrix *A, vMatrix *B, vMatrix *R) {

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                A->n_row, B->n_row, A->n_col, 1,
                A->M, A->n_col, B->M, B->n_col,
                0, R->M, B->n_row);

    // cost: A_col*B_row + 2*A_row*A_col*B_row
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //         A->n_row, B->n_row, A->n_col, 1,
    //         A->M, A->n_row, B->M, B->n_col,
    //         0, R->M, A->n_row);
    
}

//  Straightforward implementation (no BLAS)
void matrix_rtrans_mul_s(vMatrix *A, vMatrix *B, vMatrix *R) {

    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_row; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i * R->n_col + j] += A->M[i * A->n_col + k] * B->M[j * B->n_col + k];

            }
        }
    }
}

// Working impl
void v_matrix_rtrans_mul(vMatrix *A, vMatrix *B, vMatrix *R) {
    // cost: A_col*B_row + 2*A_row*A_col*B_row
    matrix_rtrans_mul_rm(A, B, R);
}
//_____________________________________________________________________________


/**
 * @brief prints the matrix
 * @param matrix    the matrix to be printed
 */
void print_v_matrix(vMatrix *matrix) {

    printf("Printing a matrix with %d rows and %d cols\n\n", matrix->n_row, matrix->n_col);
    for (int row = 0; row < matrix->n_row; row++) {
        for (int col = 0; col < matrix->n_col; col++) {
            fprintf(stdout, "%.2lf\t", matrix->M[row * matrix->n_col + col]);
        }
        fprintf(stdout, "\n\n");
    }
    fprintf(stdout, "\n\n");
}

/**
 * @brief initialize a matrix with random numbers between 0 and 1
 * @param matrix    the matrix to be initialized
 */
void random_v_matrix_init(vMatrix *matrix, double min, double max) {

    for (int i = 0; i < matrix->n_row * matrix->n_col; i++)
        matrix->M[i] = rand_from(min, max);
}

/**
 * @brief initialize a matrix W averaging columns of X
 * @param V    matrix to be factorized
 * @param W    factorizing matrix, initialized here
 * @param q    number of columns of X averaged to obtsain a column of W
 */
void random_v_acol_matrix_init(vMatrix *V, vMatrix *W, int q) {		// W->n_col * (2 * q + V->n_row * q + V->n_row)
    int r;

    // initialize W to all zeros
    memset(W->M, 0, sizeof(double) * W->n_col * W->n_row);

    for(int  k = 0; k < W -> n_col; k++){

        //average q random columns of X into W
        for (int i = 0; i < q; i++){
            r = rand() % V->n_col;					                    // 2 * W->n_col * q
            for (int j = 0; j < V -> n_row; j++)
                W->M[j * W->n_col + k] += V->M[j * V->n_col + r];   //W->M[j][k] += V->M[j][r];		
        }

        for (int j = 0; j < V -> n_row; j++)
             W->M[j * W->n_col + k] = W->M[j * W->n_col + k] / q;       //W->M[j][k] = W->M[j][k] / q;			
    }
}




/**
 * @brief computes the non-negative matrix factorisation updating the values stored by the
 *        factorization functions
 *
 * @param V     the matrix to be factorized
 * @param W     the first matrix in which V will be factorized
 * @param H     the second matrix in which V will be factorized
 */
double nnm_factorization_bs2(vMatrix *V, vMatrix *W, vMatrix *H, int maxIteration, double epsilon) {
    /**
     * @brief cost of the function
     * 
     * V = m x n
     * W = m x r
     * H = r x n
     * 
     * (3 + 7*r*n + 3*m*r + 6n*m + 10*m*r*n + 2*m*r*r + 2*r*r*n) * Number_of_iterations
     * 
     */
    int count = maxIteration;

    //Operands needed to compute Hn+1
    vMatrix numerator, denominator_l, denominator;

    numerator.n_row = W->n_col;
    numerator.n_col = V->n_col;

    denominator_l.n_row = W->n_col;
    denominator_l.n_col = W->n_col;

    denominator.n_row = H->n_row;
    denominator.n_col = H->n_col;

    v_matrix_allocation(&numerator);
    v_matrix_allocation(&denominator);
    v_matrix_allocation(&denominator_l);

    //Operands needed to compute Wn+1
    vMatrix numerator_W, denominator_l_W, denominator_W;

    numerator_W.n_row = V->n_row;
    numerator_W.n_col = H->n_row;

    denominator_l_W.n_row = W->n_row;
    denominator_l_W.n_col = H->n_col;

    denominator_W.n_row = W->n_row;
    denominator_W.n_col = W->n_col;

    v_matrix_allocation(&numerator_W);
    v_matrix_allocation(&denominator_W);
    v_matrix_allocation(&denominator_l_W);

    //real convergence computation
    double err;

    for (;;) {
        err = v_error(V, W, H);     //cost: 3 + n*r + 5*m*n + 2*m*r*n
        if (maxIteration > 0 && count == 0) {
            break;
        }
        //printf("%lf\n", err);
        if (err <= epsilon) {
            break;
        }
        count--;
        
        //printf("Current error: %lf\n", err);

        //computation for Hn+1
        v_matrix_ltrans_mul(W, V, &numerator);          // cost: B_col*B_row + 2*B_row*A_col*B_col = n*m + 2*m*r*n
        v_matrix_ltrans_mul(W, W, &denominator_l);      // cost: B_col*B_row + 2*B_row*A_col*B_col = m*r + 2*m*r*r
        v_matrix_mul(&denominator_l, H, &denominator);  // cost: B_col*A_col + 2*A_row*A_col*B_col = n*r + 2*r*r*n

        for (int i = 0; i < H->n_row * H->n_col; i++)
            H->M[i] = H->M[i] * numerator.M[i] / denominator.M[i]; // 2*r*n

        //computation for Wn+1
        v_matrix_rtrans_mul(V, H, &numerator_W);                   // cost: A_col*B_row + 2*A_row*A_col*B_row = n*r + 2*m*n*r
        v_matrix_mul(W, H, &denominator_l_W);                      // cost: B_col*A_col + 2*A_row*A_col*B_col = n*r + 2*m*r*n
        v_matrix_rtrans_mul(&denominator_l_W, H, &denominator_W);  // cost: A_col*B_row + 2*A_row*A_col*B_row = n*r + 2*m*n*r

        for (int i = 0; i < W->n_row * W->n_col; i++)
            W->M[i] = W->M[i] * numerator_W.M[i] / denominator_W.M[i]; // 2*m*r
    }

    v_matrix_deallocation(&numerator);
    v_matrix_deallocation(&denominator);
    v_matrix_deallocation(&denominator_l);
    v_matrix_deallocation(&numerator_W);
    v_matrix_deallocation(&denominator_W);
    v_matrix_deallocation(&denominator_l_W);
    return err;
}

/**
 * @brief computes the error based on the Frobenius norm 0.5*||V-WH||^2. The error is
 *        normalized with the norm V
 *
 * @param V is the original matrix
 * @param W is the first factorization matrix
 * @param H is the second factorization matrix
 * @return is the error
 */
double v_error(vMatrix *V, vMatrix *W, vMatrix *H) {

    //cost: 3 + n*r + 5*m*n + 2*m*r*n

    vMatrix approximation;

    approximation.n_row = V->n_row;
    approximation.n_col = V->n_col;

    v_matrix_allocation(&approximation);

    v_matrix_mul(W, H, &approximation);  // cost: B_col*A_col + 2*A_row*A_col*B_col = n*r + 2*m*r*n 

    double V_norm = v_norm(V); // cost: 2 * matrix_row * matrix_col + 1 = 2*m*n + 1
    double approximation_norm;


    for (int i = 0; i < V->n_row * V->n_col; i++)
        approximation.M[i] = (V->M[i] - approximation.M[i]); // cost: n*m

    approximation_norm = v_norm(&approximation); // cost: 2 * matrix_row * matrix_col + 1 = 2*m*n + 1

    v_matrix_deallocation(&approximation);

    return approximation_norm / V_norm; //cost: 1
}


/**
 * @brief computes the frobenius norm of a matrix
 *
 * @param matrix is the matrix which norm is computed
 * @return the norm
 */
double v_norm(vMatrix *matrix) {

    // cost: 2 * matrix_row * matrix_col + 1
    double temp_norm = 0;
    for (int i = 0; i < matrix->n_row * matrix->n_col; i++)
        temp_norm += matrix->M[i] * matrix->M[i];

    return sqrt(temp_norm);
}



/* --------------- COST FUNCTIONS ---------------- */
/**
 * @brief returns the cost of the function nnm_factorization_bs1
 * @param V     the matrix to be factorized
 * @param W     the first matrix in which V will be factorized
 * @param H     the second matrix in which V will be factorized
 */
double nnm_factorization_bs2_cost(vMatrix *V, vMatrix *W, vMatrix *H, int numIterations) {
    
    unsigned long long n, m, r;
    m = V->n_row;
    n = V->n_col;
    r = W->n_col;
    return (double) (3 + 7*r*n + 3*m*r + 6*n*m + 10*m*r*n + 2*m*r*r + 2*r*r*n) * numIterations;
}

/**
 * @brief returns the cost of the funcion random_acol_matrix_init
 * @param V    matrix to be factorized
 * @param W    factorizing matrix, initialized here
 * @param q    number of columns of X averaged to obtsain a column of W
 */
long random_acol_v_matrix_init_cost(vMatrix* V, Matrix* W, int q)
{
    return W->n_col * (2 * q + V->n_row * q + V->n_row);
}

/**
 * @brief returns the cost of the funcion random_matrix_init
 * @param matrix    the matrix to be initialized, containing the info on its size
 */
long random_v_matrix_init_cost(vMatrix* matrix)
{
    return 5 * matrix->n_row * matrix->n_col;
}

