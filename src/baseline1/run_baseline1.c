#include "baseline1.h"

/**
 * @brief represents a dynamic allocated matrix
 * @param M     is the matrix
 * @param n_row is the number of rows
 * @param n_col is the number of cols
 */

int main(int argc, char const *argv[]) {

    Matrix V;
    Matrix W, H;
    int m, n, r;

    srand(time(NULL));
    // read the desired factorization dimension
    fscanf(stdin, "%d", &r);
    // read the dimensions
    fscanf(stdin, "%d %d", &m, &n);

    V.n_row = m;
    V.n_col = n;
    matrix_allocation(&V);

    W.n_row = m;
    W.n_col = r;
    matrix_allocation(&W);

    H.n_row = r;
    H.n_col = n;
    matrix_allocation(&H);

    random_matrix_init(&W, 0, 1);					// 5 * W->n_row * W->n_col
    random_matrix_init(&H, 0, 1);					// 5 * H->n_row * H->n_col

    read_input(&V);
    print_matrix(&V);

    nnm_factorization_bs1(&V, &W, &H, 100, 0.5);	// go to baseline1.c nnm_factorization_bs1 for cost

    print_matrix(&W);
    print_matrix(&H);

    matrix_deallocation(&V);
    matrix_deallocation(&W);
    matrix_deallocation(&H);

    return 0;
}