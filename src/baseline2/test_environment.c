//
// Created by Viktor Gsteiger on 08.03.22.
//
#include <stdlib.h>
#include <stdio.h>
#include "cblas.h"
#include "math.h"

typedef struct {
    double *M;
    int n_row;
    int n_col;
} Matrix;

double rand_from(double min, double max) {

    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void matrix_allocation(Matrix *matrix) {

    // allocate the matrix dynamically
    matrix->M = malloc(sizeof(double *) * matrix->n_row * matrix->n_col);
}

void random_matrix_init(Matrix *matrix) {

    for (int i = 0; i < matrix->n_row * matrix->n_col; i++)
        matrix->M[i] = rand_from(0.00, 1.00);
}

void matrix_mul_cblas(Matrix *A, Matrix *B, Matrix *R) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A->n_row, B->n_col, A->n_col, 1,
                A->M, A->n_col, B->M, B->n_col,
                0, R->M, B->n_col);
}

void matrix_mul_straight(Matrix *A, Matrix *B, Matrix *R) {
    for (int i = 0; i < A->n_row; i++) {
        for (int j = 0; j < B->n_col; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i * R->n_col + j] += A->M[i * A->n_col + k] * B->M[k * B->n_col + j];

            }

        }
    }
}

void matrix_mult_left(Matrix *A, Matrix *B, Matrix *R) {
    for (int i = 0; i < A->n_row; i++)
        for (int j = 0; j < B->n_row; j++) {
            R->M[i * R->n_col + j] = 0;
            for (int k = 0; k < A->n_col; k++) {
                R->M[i * R->n_col + j] += A->M[i * A->n_col + k] * B->M[j * B->n_col + k];
            }
        }
}

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

void compare_mat(Matrix *A, Matrix *B) {
    for (int row = 0; row < A->n_row; row++) {
        for (int col = 0; col < A->n_col; col++) {
            if (floor(10000 * A->M[row * A->n_col + col]) / 10000 !=
                floor(10000 * B->M[row * B->n_col + col]) / 10000) {
                printf("(%f, %f)", A->M[row * A->n_col + col], B->M[row * B->n_col + col]);
                printf("Not the same, boho!");
                return;
            }
        }
    }
    printf("The same, hell yeah!");
}

int main(int argc, char const *argv[]) {
    for (int i = 5; i < 1000; i += 100) {
        Matrix V1, V2;
        Matrix W, H;
        int m = i, n = i + 2, r = i + 3;

        V1.n_row = m;
        V1.n_col = n;
        matrix_allocation(&V1);

        V2.n_row = m;
        V2.n_col = n;
        matrix_allocation(&V2);

        W.n_row = m;
        W.n_col = r;
        matrix_allocation(&W);

        H.n_row = r;
        H.n_col = n;
        matrix_allocation(&H);

        random_matrix_init(&W);
        random_matrix_init(&H);

        random_matrix_init(&V1);
        random_matrix_init(&V2);

        matrix_mul_straight(&W, &H, &V1);
        //print_matrix(&V1);

        matrix_mul_cblas(&W, &H, &V2);
        //print_matrix(&V2);

        printf("Running %i: \n", i);
        compare_mat(&V1, &V2);
        printf("\n");
    }
}


