#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "baseline1.h"

void read_input(Matrix *matrix);

void random_acol_matrix_init(Matrix *V, Matrix *W, int q);

void matrix_mul(Matrix *A, Matrix *B, Matrix *R);

void matrix_ltrans_mul(Matrix *A, Matrix *B, Matrix *R);

void matrix_rtrans_mul(Matrix *A, Matrix *B, Matrix *R);

void print_matrix(Matrix *matrix);

double error(Matrix *V, Matrix *W, Matrix *H);

double rand_from(double min, double max);

double norm(Matrix *matrix);

/**
 * @brief reads the input corresponfind to the matrix values
 * @param matrix    the matrix that will be filled
 */
void read_input(Matrix *matrix) {

    for (int row = 0; row < matrix->n_row; row++) {
        for (int col = 0; col < matrix->n_col; col++)
            fscanf(stdin, "%lf", &(matrix->M[row][col]));
    }
}

/**
 * @brief compute the multiplication of A and B
 * @param A is the first factor 
 * @param B is the other factor of the multiplication
 * @param R is the matrix that will hold the result
 */
void matrix_mul(Matrix *A, Matrix *B, Matrix *R) {			// 2 * A->n_row * B->n_col * A->n_col

    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i][j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i][j] += A->M[i][k] * B->M[k][j];
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
void matrix_ltrans_mul(Matrix *A, Matrix *B, Matrix *R) {	// 2 * A->n_col * B->n_col * B->n_row

    for (int i = 0; i < A->n_col; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i][j] = 0;
            for (int k = 0; k < B->n_row; k++) {
                R->M[i][j] += A->M[k][i] * B->M[k][j];

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
void matrix_rtrans_mul(Matrix *A, Matrix *B, Matrix *R) {	// 2 * A->n_row * B->n_row * A->n_col

    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_row; j++) {
            R->M[i][j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i][j] += A->M[i][k] * B->M[j][k];
            }
        }
    }
}

/**
 * @brief prints the matrix
 * @param matrix    the matrix to be printed
 */
void print_matrix(Matrix *matrix) {

    printf("Printing a matrix with %d rows and %d cols\n\n", matrix->n_row, matrix->n_col);
    for (int row = 0; row < matrix->n_row; row++) {
        for (int col = 0; col < matrix->n_col; col++) {
            fprintf(stdout, "%.2lf\t", matrix->M[row][col]);
        }
        fprintf(stdout, "\n\n");
    }
    fprintf(stdout, "\n\n");
}

/**
 * @brief initialize a matrix with random numbers between 0 and 1
 * @param matrix    the matrix to be initialized
 */
void random_matrix_init(Matrix *matrix, double min, double max) {	// 5 * matrix->n_row * matrix->n_col

    for (int row = 0; row < matrix->n_row; row++) {
        for (int col = 0; col < matrix->n_col; col++) {
            matrix->M[row][col] = rand_from(min, max);
        }
    }
}


/**
 * @brief initialize a matrix W averaging columns of X
 * @param V    matrix to be factorized
 * @param W    factorizing matrix, initialized here
 * @param q    number of columns of X averaged to obtsain a column of W
 */
void random_acol_matrix_init(Matrix *V, Matrix *W, int q) {		// W->n_col * (2 * q + V->n_row * q + V->n_row)
    int r;

    // initialize W to all zeros
    for(int k = 0; k < W -> n_row; k++)
        memset(W->M[k], 0,  sizeof(double) * W->n_col);

    for(int  k = 0; k < W -> n_col; k++){
        //average q random column of X into W

        for (int i = 0; i < q; i++){
            r = rand() % V->n_col;					// 2 * W->n_col * q
            for (int j = 0; j < V -> n_row; j++)
                W->M[j][k] += V->M[j][r];			// W->n_col * q * V->n_row
        }

        for (int j = 0; j < V -> n_row; j++)
            W->M[j][k] = W->M[j][k] / q;			// W->n_col * V->n_row
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
double nnm_factorization_bs1(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon) {
	// 2 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col + 3 +
	// i * (4 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col +
	// 		2 * W->n_col * V->n_col * V->n_row +
	//		2 * W->n_col * W->n_col * W->n_row +
	//		2 * W->n_row * H->n_col * H->n_col +
	//		2 * V->n_row * H->n_row * V->n_col +
	//		2 * W->n_row * H->n_row * H->n_col +
	//		2 * H->n_row * H->n_col +
	//		2 * W->n_row * W->n_col + 3)
	
    int count = maxIteration;

    //Operands needed to compute Hn+1
    Matrix numerator, denominator_l, denominator;

    numerator.n_row = W->n_col;
    numerator.n_col = V->n_col;

    denominator_l.n_row = W->n_col;
    denominator_l.n_col = W->n_col;

    denominator.n_row = H->n_row;
    denominator.n_col = H->n_col;

    matrix_allocation(&numerator);
    matrix_allocation(&denominator);
    matrix_allocation(&denominator_l);

    //Operands needed to compute Wn+1
    Matrix numerator_W, denominator_l_W, denominator_W;

    numerator_W.n_row = V->n_row;
    numerator_W.n_col = H->n_row;

    denominator_l_W.n_row = W->n_row;
    denominator_l_W.n_col = H->n_col;

    denominator_W.n_row = W->n_row;
    denominator_W.n_col = W->n_col;

    matrix_allocation(&numerator_W);
    matrix_allocation(&denominator_W);
    matrix_allocation(&denominator_l_W);

    //real convergence computation
    double err;											
    err = error(V, W, H);								// 2 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col + 3
    for (;;) {											// i * (4 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col +
														// 		2 * W->n_col * V->n_col * V->n_row +
														//		2 * W->n_col * W->n_col * W->n_row +
														//		2 * W->n_row * H->n_col * H->n_col +
														//		2 * V->n_row * H->n_row * V->n_col +
														//		2 * W->n_row * H->n_row * H->n_col +
														//		2 * H->n_row * H->n_col +
														//		2 * W->n_row * W->n_col + 3)
        if (maxIteration > 0 && count == 0) {
            break;
        }
        if (err <= epsilon) {							// *** is comparison an op
            break;
        }
        count--;
        err = error(V, W, H);							// 2 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col + 3 **
        //printf("Current error: %lf\n", err);

        //computation for Hn+1
        matrix_ltrans_mul(W, V, &numerator);			// 2 * W->n_col * V->n_col * V->n_row **
        matrix_ltrans_mul(W, W, &denominator_l);		// 2 * W->n_col * W->n_col * W->n_row **
        matrix_mul(&denominator_l, H, &denominator);	// 2 * W->n_row * H->n_col * H->n_col **

        for (int i = 0; i < H->n_row; i++) {			// 2 * H->n_row * H->n_col **
            for (int j = 0; j < H->n_col; j++) {
                H->M[i][j] = H->M[i][j] * numerator.M[i][j] / denominator.M[i][j];
            }
        }

        //computation for Wn+1
        matrix_rtrans_mul(V, H, &numerator_W);					// 2 * V->n_row * H->n_row * V->n_col **
        matrix_mul(W, H, &denominator_l_W);						// 2 * W->n_row * H->n_col * W->n_col **
        matrix_rtrans_mul(&denominator_l_W, H, &denominator_W);	// 2 * W->n_row * H->n_row * H->n_col **

        for (int i = 0; i < W->n_row; i++) {					// 2 * W->n_row * W->n_col **
            for (int j = 0; j < W->n_col; j++) {
                W->M[i][j] = W->M[i][j] * numerator_W.M[i][j] / denominator_W.M[i][j];
            }
        }
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
 * @brief computes the error based on the Frobenius norm 0.5*||V-WH||^2. The error is
 *        normalized with the norm V
 * 
 * @param V is the original matrix
 * @param W is the first factorization matrix
 * @param H is the second factorization matrix
 * @return is the error
 */
double error(Matrix *V, Matrix *W, Matrix *H) {		// 2 * W->n_row * H->n_col * W->n_col + 5 * V->n_row * V->n_col + 3

    Matrix approximation;

    approximation.n_row = V->n_row;
    approximation.n_col = V->n_col;

    matrix_allocation(&approximation);
    matrix_mul(W, H, &approximation);				//cost of mul --- 2 * W->n_row * H->n_col * W->n_col

    double V_norm = norm(V);						// 2 * V->n_row * V->n_col + 1
    double approximation_norm;

    for (int row = 0; row < V->n_row; row++)
        for (int col = 0; col < V->n_col; col++) {
            approximation.M[row][col] = (V->M[row][col] - approximation.M[row][col]);  // V->n_row * V->n_col
        }

    approximation_norm = norm(&approximation);		// 2 * V->n_row * V->n_col + 1
    matrix_deallocation(&approximation);
    return approximation_norm / V_norm;				// 1
}


/**
 * @brief computes the frobenius norm of a matrix
 * 
 * @param matrix is the matrix which norm is computed
 * @return the norm 
 */
double norm(Matrix *matrix) {		// 2 * matrix->n_row * matrix->n_col + 1

    double temp_norm = 0;

    for (int row = 0; row < matrix->n_row; row++) {
        for (int col = 0; col < matrix->n_col; col++) {
            temp_norm += matrix->M[row][col] * matrix->M[row][col];
        }
    }

    return sqrt(temp_norm);
}
