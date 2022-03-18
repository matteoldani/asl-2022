#include "baseline2.h"

void print_matrix(vMatrix *matrix);

/**
 * @brief prints the matrix
 * @param matrix    the matrix to be printed
 */
void print_matrix(vMatrix *matrix) {

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
 * @brief represents a dynamic allocated matrix
 * @param M     is the matrix
 * @param n_row is the number of rows
 * @param n_col is the number of cols
 */

int main(int argc, char const *argv[]) {

    vMatrix V;
    vMatrix W, H;
    int m = 100, n = 110, r = 12;

    srand(time(NULL));
    // read the desired factorization dimension
    //fscanf(stdin, "%d", &r);
    // read the dimensions
    //fscanf(stdin, "%d %d", &m, &n);

    V.n_row = m;
    V.n_col = n;
    v_matrix_allocation(&V);

    W.n_row = m;
    W.n_col = r;
    v_matrix_allocation(&W);

    H.n_row = r;
    H.n_col = n;
    v_matrix_allocation(&H);

    random_v_matrix_init(&W, 0, 1);
    random_v_matrix_init(&H, 0, 1);

    random_v_matrix_init(&V, 0, 1);
    print_matrix(&V);

    nnm_factorization_bs2(&V, &W, &H, 100, 0.5);

    print_matrix(&W);
    print_matrix(&H);

    v_matrix_deallocation(&V);
    v_matrix_deallocation(&W);
    v_matrix_deallocation(&H);

    return 0;
}