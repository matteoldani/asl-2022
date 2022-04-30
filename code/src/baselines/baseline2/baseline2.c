#include <baselines/baseline2.h>

#if __APPLE__
#include <Accelerate/Accelerate.h>
#else

#include "cblas.h"

#endif

// _____________________________ MATRIX MUL _____________________________
/**
 * @brief compute the multiplication of A and B
 * @param A is the first factor
 * @param B is the other factor of the multiplication
 * @param R is the matrix that will hold the result
 */

// RowMajor implementation
void matrix_mul_bs2_rm(Matrix *A, Matrix *B, Matrix *R) {

    // cost: B_col*A_col + 2*A_row*A_col*B_col
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A->n_row, B->n_col, A->n_col, 1,
                A->M, A->n_col, B->M, B->n_col,
                0, R->M, B->n_col);
}

// Straightforward implementation (no BLAS)
void matrix_mul_bs2_s(Matrix *A, Matrix *B, Matrix *R) {

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
void matrix_mul_bs2(Matrix *A, Matrix *B, Matrix *R) {
     // cost: B_col*A_col + 2*A_row*A_col*B_col
    matrix_mul_bs2_rm(A, B, R);
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
void matrix_ltrans_mul_bs2_cm(Matrix *A, Matrix *B, Matrix *R) {

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                A->n_col, B->n_col, A->n_row, 1,
                A->M, A->n_col, B->M, B->n_row,
                0, R->M, A->n_col);
}

// RowMajor impl (0.47)
void matrix_ltrans_mul_bs2_rm(Matrix *A, Matrix *B, Matrix *R) {
     // cost: B_col*B_row + 2*B_row*A_col*B_col
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                A->n_col, B->n_col, B->n_row, 1, //r=A->n_row = B->n_row
                A->M, A->n_col, B->M, B->n_col,
                0, R->M, B->n_col);
}

// Straightforward implementation (no BLAS)
void matrix_ltrans_mul_bs2_s(Matrix *A, Matrix *B, Matrix *R) {

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
void matrix_ltrans_mul_bs2(Matrix *A, Matrix *B, Matrix *R) {
   
    matrix_ltrans_mul_bs2_rm(A, B, R);
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
void matrix_rtrans_mul_bs2_cm(Matrix *A, Matrix *B, Matrix *R) {

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                A->n_row, B->n_row, A->n_col, 1,
                A->M, A->n_row, B->M, B->n_col,
                0, R->M, A->n_row);
}

// RowMajor impl  ----Param num 10 has an illegal value (0.475054)
void matrix_rtrans_mul_bs2_rm(Matrix *A, Matrix *B, Matrix *R) {

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
void matrix_rtrans_mul_bs2_s(Matrix *A, Matrix *B, Matrix *R) {

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
void matrix_rtrans_mul_bs2(Matrix *A, Matrix *B, Matrix *R) {
    // cost: A_col*B_row + 2*A_row*A_col*B_row
    matrix_rtrans_mul_bs2_rm(A, B, R);
}
//_____________________________________________________________________________



/**
 * @brief computes the non-negative matrix factorisation updating the values stored by the
 *        factorization functions
 *
 * @param V     the matrix to be factorized
 * @param W     the first matrix in which V will be factorized
 * @param H     the second matrix in which V will be factorized
 */
double nnm_factorization_bs2(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon) {
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

    //Operands needed to compute Hn+1
    Matrix numerator, denominator_l, denominator;

    matrix_allocation(&numerator, W->n_col, V->n_col);
    matrix_allocation(&denominator, H->n_row, H->n_col);
    matrix_allocation(&denominator_l, W->n_col, W->n_col);

    //Operands needed to compute Wn+1
    Matrix numerator_W, denominator_l_W, denominator_W;

    matrix_allocation(&numerator_W, V->n_row, H->n_row);
    matrix_allocation(&denominator_W, W->n_row, W->n_col);
    matrix_allocation(&denominator_l_W, W->n_row, H->n_col);

    //real convergence computation
    double err = -1;

    for (int count = 0; count<maxIteration; count++) {
        err = error_bs2(V, W, H);     //cost: 3 + n*r + 5*m*n + 2*m*r*n

        //printf("%lf\n", err);
        if (err <= epsilon) {
            break;
        }
        
        //printf("Current error_bs2: %lf\n", err);

        //computation for Hn+1
        matrix_ltrans_mul_bs2(W, V, &numerator);          // cost: B_col*B_row + 2*B_row*A_col*B_col = n*m + 2*m*r*n
        matrix_ltrans_mul_bs2(W, W, &denominator_l);      // cost: B_col*B_row + 2*B_row*A_col*B_col = m*r + 2*m*r*r
        matrix_mul_bs2(&denominator_l, H, &denominator);  // cost: B_col*A_col + 2*A_row*A_col*B_col = n*r + 2*r*r*n

        for (int i = 0; i < H->n_row * H->n_col; i++)
            H->M[i] = H->M[i] * numerator.M[i] / denominator.M[i]; // 2*r*n

        //computation for Wn+1
        matrix_rtrans_mul_bs2(V, H, &numerator_W);                   // cost: A_col*B_row + 2*A_row*A_col*B_row = n*r + 2*m*n*r
        matrix_mul_bs2(W, H, &denominator_l_W);                      // cost: B_col*A_col + 2*A_row*A_col*B_col = n*r + 2*m*r*n
        matrix_rtrans_mul_bs2(&denominator_l_W, H, &denominator_W);  // cost: A_col*B_row + 2*A_row*A_col*B_row = n*r + 2*m*n*r

        for (int i = 0; i < W->n_row * W->n_col; i++)
            W->M[i] = W->M[i] * numerator_W.M[i] / denominator_W.M[i]; // 2*m*r
    }

    matrix_deallocation(&numerator);
    matrix_deallocation(&denominator);
    matrix_deallocation(&denominator_l);
    matrix_deallocation(&numerator_W);
    matrix_deallocation(&denominator_W);
    matrix_deallocation(&denominator_l_W);
    return err;
}

/**
 * @brief computes the error_bs2 based on the Frobenius norm 0.5*||V-WH||^2. The error_bs2 is
 *        normalized with the norm V
 *
 * @param V is the original matrix
 * @param W is the first factorization matrix
 * @param H is the second factorization matrix
 * @return is the error_bs2
 */
double error_bs2(Matrix *V, Matrix *W, Matrix *H) {

    //cost: 3 + n*r + 5*m*n + 2*m*r*n

    Matrix approximation;

    matrix_allocation(&approximation, V->n_row, V->n_col);

    matrix_mul_bs2(W, H, &approximation);  // cost: B_col*A_col + 2*A_row*A_col*B_col = n*r + 2*m*r*n 

    double V_norm = norm(V); // cost: 2 * matrix_row * matrix_col + 1 = 2*m*n + 1
    double approximation_norm;


    for (int i = 0; i < V->n_row * V->n_col; i++)
        approximation.M[i] = (V->M[i] - approximation.M[i]); // cost: n*m

    approximation_norm = norm(&approximation); // cost: 2 * matrix_row * matrix_col + 1 = 2*m*n + 1

    matrix_deallocation(&approximation);

    return approximation_norm / V_norm; //cost: 1
}




/* --------------- COST FUNCTIONS ---------------- */
/**
 * @brief returns the cost of the function nnm_factorization_bs1
 * @param V     the matrix to be factorized
 * @param W     the first matrix in which V will be factorized
 * @param H     the second matrix in which V will be factorized
 */
myInt64 nnm_factorization_bs2_cost(Matrix *V, Matrix *W, Matrix *H, int numIterations) {
    
    myInt64 n, m, r;
    m = V->n_row;
    n = V->n_col;
    r = W->n_col;
    return (myInt64) (3 + 7*r*n + 3*m*r + 6*n*m + 10*m*r*n + 2*m*r*r + 2*r*r*n) * numIterations;
}

/**
 * @brief returns the cost of the funcion random_acol_matrix_init
 * @param V    matrix to be factorized
 * @param W    factorizing matrix, initialized here
 * @param q    number of columns of X averaged to obtsain a column of W
 */
myInt64 random_acol_v_matrix_init_cost(Matrix* V, Matrix* W, int q)
{
    return W->n_col * (2 * q + V->n_row * q + V->n_row);
}

/**
 * @brief returns the cost of the funcion random_matrix_init
 * @param matrix    the matrix to be initialized, containing the info on its size
 */
myInt64 random_v_matrix_init_cost(Matrix* matrix)
{
    return 5 * matrix->n_row * matrix->n_col;
}

