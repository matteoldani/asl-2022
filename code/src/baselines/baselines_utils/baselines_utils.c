#include <baselines/baselines_utils.h>


myInt64 matrix_mul_cost(int n, int m, int r){
    return (myInt64)(2 * n * m * r);
}

myInt64 nnm_cost(int V_row, int V_col, int W_row, int W_col, int H_row, int H_col, int num_iterations){

    return (myInt64)(2 * W_row * H_col * W_col + 5 * V_row * V_col + 3) +
           (myInt64)num_iterations * (myInt64) (4 * W_row * H_col * W_col + 5 * V_row * V_col +
                            2 * W_col * V_col * V_row +
                            2 * W_col * W_col * W_row +
                            2 * W_col * W_col * H_col +
                            2 * V_row * H_row * V_col +
                            2 * W_row * H_row * H_col +
                            2 * H_row * H_col +
                            2 * W_row * W_col + 3);
}

myInt64 matrix_rand_init_cost(int row, int col){
    return (myInt64) (5 * row * col);
}

/**
 * @brief generate a random floating point number from min to max
 * @param min   the minumum possible value
 * @param max   the maximum possible value
 * @return      the random value
 */
double rand_from(double min, double max) {	// 5

    double range = (max - min);		// 1
    double div = RAND_MAX / range;	// 1
    return min + (rand() / div);	// 3
}


/**
 * @brief allocates the matrix as an array inside the struct
 * @param matrix    is the struct where the matrix will be allocated
 * @param rows      the number of rows
 * @param cols      the number of cols          
 */
void matrix_allocation(Matrix *matrix, int rows, int cols) {
    // allocate the matrix dynamically
    matrix->M = malloc(sizeof(double *) * rows * cols);
    matrix->n_row = rows;
    matrix->n_col = cols;
}

/**
 * @brief deallocates the matrix
 * @param matrix    is the struct where the matrix will be deallocated
 */
void matrix_deallocation(Matrix *matrix) {
    free(matrix->M);
}

/**
 * @brief Allocates all the base matrices needed for the computation
 * 
 * @param matrices Set of 3 matrices to be allocated
 * @param m Rows of the matrix to be factorized
 * @param n Cols of the matrix to be factotized
 * @param r Factorization param
 */
void allocate_base_matrices(Matrices *matrices, int m, int n, int r) {

    matrix_allocation(&matrices->V, m, n);
    matrix_allocation(&matrices->W, m, r);
    matrix_allocation(&matrices->H, r, n);
}


/**
 * @brief initialize a matrix with random numbers between 0 and 1
 * @param matrix    the matrix to be initialized
 */
void random_matrix_init(Matrix *matrix, double min, double max) {

    for (int i = 0; i < matrix->n_row * matrix->n_col; i++)
        matrix->M[i] = rand_from(min, max);
}

/**
 * @brief initialize a matrix W averaging columns of X
 * @param V    matrix to be factorized
 * @param W    factorizing matrix, initialized here
 * @param q    number of columns of X averaged to obtsain a column of W
 */
void random_acol_matrix_init(Matrix *V, Matrix *W, int q) {		// W_col * (2 * q + V_row * q + V_row)

    int r;

    // initialize W to all zeros
    memset(W->M, 0, sizeof(double) * W->n_col * W->n_row);

    for(int  k = 0; k < W -> n_col; k++){

        //average q random columns of X into W
        for (int i = 0; i < q; i++){
            r = rand() % V->n_col;					                    // 2 * W_col * q
            for (int j = 0; j < V -> n_row; j++)
                W->M[j * W->n_col + k] += V->M[j * V->n_col + r];   //W->M[j][k] += V->M[j][r];		
        }

        for (int j = 0; j < V -> n_row; j++)
             W->M[j * W->n_col + k] = W->M[j * W->n_col + k] / q;       //W->M[j][k] = W->M[j][k] / q;			
    }
}

/**
 * @brief computes the frobenius norm of a matrix
 *
 * @param matrix is the matrix which norm is computed
 * @return the norm
 */
double norm(Matrix *matrix) {

    // cost: 2 * matrix_row * matrix_col + 1
    double temp_norm = 0;
    for (int i = 0; i < matrix->n_row * matrix->n_col; i++)
        temp_norm += matrix->M[i] * matrix->M[i];

    return sqrt(temp_norm);
}


/**
 * @brief prints the matrix
 * @param matrix    the matrix to be printed
 */
void print_matrix(Matrix *matrix) {

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
 * @brief   allocates the correspondin matrix 
 *          and fill the matrix with the values 
 *          read from the file
 * 
 * @param matrix    the matrix that will be filled
 * @param r         is the factorization parameter
 * @param file      is the file to read the matrix from
 */
void allocate_from_file(Matrix *matrix, int *r, FILE *file) {

    int n_row;
    int n_col;

    printf("Reading matrix information: \n");
    
    fscanf(file, "%d", r);
    printf("\tr: %d\n", *r);
    fscanf(file, "%d", &n_row);
    printf("\tn_row: %d\n", n_row);
    fscanf(file, "%d", &n_col);
    printf("\tn_col: %d\n", n_col);

    matrix_allocation(matrix, n_row, n_col);

    for (int i=0; i<matrix->n_row * matrix->n_col; i++){
        fscanf(file, "%lf", &(matrix->M[i]));
    }

    
}