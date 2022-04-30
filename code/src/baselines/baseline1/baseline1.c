#include <baselines/baseline1.h>


/**
 * @brief compute the multiplication of A and B
 * @param A is the first factor 
 * @param B is the other factor of the multiplication
 * @param R is the matrix that will hold the result
 */
void matrix_mul_bs1(Matrix *A, Matrix *B, Matrix *R) {			// 2 * A->n_row * B->n_col * A->n_col

    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i * R->n_col + j] += A->M[i * A->n_col + k] * B->M[k * B->n_col + j];
            }
        }
    }
}

/**
 * @brief compute the multiplication of A^T and B
 * @param A is the matrix to be transposed
 * @param B is the other factor of the multiplication
 * @param R is the matrix that will hold the result
 */
void matrix_ltrans_mul_bs1(Matrix *A, Matrix *B, Matrix *R) {	// 2 * A->n_col * B->n_col * B->n_row

    for (int i = 0; i < A->n_col; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < B->n_row; k++) {
                R->M[i * R->n_col + j] += A->M[k * A->n_col + i] * B->M[k * B->n_col + j];

            }
        }
    }
}

/**
 * @brief compute the multiplication of A and B^T
 * @param A is the other factor of the multiplication
 * @param B is the matrix to be transposed
 * @param R is the matrix that will hold the result
 */
void matrix_rtrans_mul_bs1(Matrix *A, Matrix *B, Matrix *R) {	// 2 * A->n_row * B->n_row * A->n_col

    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_row; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i * R->n_col + j] += A->M[i * A->n_col + k] * B->M[j * B->n_col + k];
            }
        }
    }
}


/**
 * @brief returns the cost of the funcion random_matrix_init
 * @param matrix    the matrix to be initialized, containing the info on its size
 */
myInt64 random_matrix_init_cost(Matrix* matrix)
{
    return 5 * matrix->n_row * matrix->n_col;
}


/**
 * @brief returns the cost of the funcion random_acol_matrix_init
 * @param V    matrix to be factorized
 * @param W    factorizing matrix, initialized here
 * @param q    number of columns of X averaged to obtsain a column of W
 */
myInt64 random_acol_matrix_init_cost(Matrix* V, Matrix* W, int q)
{
    return W->n_col * (2 * q + V->n_row * q + V->n_row);
}



/**
 * @brief computes the non-negative matrix factorisation updating the values stored by the 
 *        factorization functions
 * 
 * @param V     the matrix to be factorized
 * @param W     the first matrix in which V will be factorized
 * @param H     the second matrix in which V will be factorized
 */
double nnm_factorization_bs1(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon) {
	// 2 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col + 3 +
	// i * (4 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col +
	// 		2 * W->n_col * V->n_col * V->n_row +
	//		2 * W->n_col * W->n_col * W->n_row +
	//		2 * W->n_col * W->n_col * H->n_col +
	//		2 * V->n_row * H->n_row * V->n_col +
	//		2 * W->n_row * H->n_row * H->n_col +
	//		2 * H->n_row * H->n_col +
	//		2 * W->n_row * W->n_col + 3)
	

    //Operands needed to compute Hn+1
    Matrix numerator, denominator_l, denominator;

    matrix_allocation(&numerator, W->n_col, V->n_col);
    matrix_allocation(&denominator_l, W->n_col, W->n_col);
    matrix_allocation(&denominator, H->n_row, H->n_col);

    //Operands needed to compute Wn+1
    Matrix numerator_W, denominator_l_W, denominator_W;

    matrix_allocation(&numerator_W, V->n_row, H->n_row);
    matrix_allocation(&denominator_W, W->n_row, W->n_col);
    matrix_allocation(&denominator_l_W, W->n_row, H->n_col);

    //real convergence computation
    double err = -1;											
    for (int count = 0; count < maxIteration; count++) {											// i * (4 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col +
														// 		2 * W->n_col * V->n_col * V->n_row +
														//		2 * W->n_col * W->n_col * W->n_row +
														//		2 * W->n_col * W->n_col * H->n_col +
														//		2 * V->n_row * H->n_row * V->n_col +
														//		2 * W->n_row * H->n_row * H->n_col +
														//		2 * H->n_row * H->n_col +
														//		2 * W->n_row * W->n_col + 3)
        
        err = error_bs1(V, W, H);
        if (err <= epsilon) {							// *** is comparison an op
            break;
        }
 
        							// 2 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col + 3 **
        //printf("Current error_bs1: %lf\n", err);

        //computation for Hn+1
        matrix_ltrans_mul_bs1(W, V, &numerator);			// 2 * W->n_col * V->n_col * V->n_row **
        matrix_ltrans_mul_bs1(W, W, &denominator_l);		// 2 * W->n_col * W->n_col * W->n_row **
        matrix_mul_bs1(&denominator_l, H, &denominator);	// 2 * W->n_col * W->n_col * H->n_col **

        for (int i = 0; i < H->n_row; i++) {			// 2 * H->n_row * H->n_col **
            for (int j = 0; j < H->n_col; j++) {
                H->M[i * H->n_col + j] = H->M[i * H->n_col + j] * numerator.M[i * numerator.n_col + j] / denominator.M[i * denominator.n_col + j];
            }
        }

        //computation for Wn+1
        matrix_rtrans_mul_bs1(V, H, &numerator_W);					// 2 * V->n_row * H->n_row * V->n_col **
        matrix_mul_bs1(W, H, &denominator_l_W);						// 2 * W->n_row * H->n_col * W->n_col **
        matrix_rtrans_mul_bs1(&denominator_l_W, H, &denominator_W);	// 2 * W->n_row * H->n_row * H->n_col **

        for (int i = 0; i < W->n_row; i++) {					// 2 * W->n_row * W->n_col **
            for (int j = 0; j < W->n_col; j++) {
                W->M[i * W->n_col + j] = W->M[i * W->n_col + j] * numerator_W.M[i * numerator_W.n_col + j] / denominator_W.M[i * denominator_W.n_col + j];
            }
        }
    //printf("Baseline 1 err: %lf\n", err);
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
 * @brief returns the cost of the function nnm_factorization_bs1
 * @param V     the matrix to be factorized
 * @param W     the first matrix in which V will be factorized
 * @param H     the second matrix in which V will be factorized
 */
myInt64 nnm_factorization_bs1_cost(Matrix* V, Matrix* W, Matrix* H, int numIterations) {
    return (myInt64)(2 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col + 3) +
           (myInt64)numIterations * (myInt64) (4 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col +
                            2 * W->n_col * V->n_col * V->n_row +
                            2 * W->n_col * W->n_col * W->n_row +
                            2 * W->n_col * W->n_col * H->n_col +
                            2 * V->n_row * H->n_row * V->n_col +
                            2 * W->n_row * H->n_row * H->n_col +
                            2 * H->n_row * H->n_col +
                            2 * W->n_row * W->n_col + 3);
}


/**
 * @brief computes the error_bs1 based on the Frobenius norm 0.5*||V-WH||^2. The error_bs1 is
 *        normalized with the norm V
 *
 * @param V is the original matrix
 * @param W is the first factorization matrix
 * @param H is the second factorization matrix
 * @return is the error_bs1
 */
double error_bs1(Matrix *V, Matrix *W, Matrix *H) {

    //cost: 3 + n*r + 5*m*n + 2*m*r*n

    Matrix approximation;

    matrix_allocation(&approximation, V->n_row, V->n_col);

    matrix_mul_bs1(W, H, &approximation);  // cost: B_col*A_col + 2*A_row*A_col*B_col = n*r + 2*m*r*n 

    double V_norm = norm(V); // cost: 2 * matrix_row * matrix_col + 1 = 2*m*n + 1
    double approximation_norm;

    for (int i = 0; i < V->n_row * V->n_col; i++)
        approximation.M[i] = (V->M[i] - approximation.M[i]); // cost: n*m

    approximation_norm = norm(&approximation); // cost: 2 * matrix_row * matrix_col + 1 = 2*m*n + 1

    matrix_deallocation(&approximation);

    return approximation_norm / V_norm; //cost: 1
}
