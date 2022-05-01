//
// Created by Viktor Gsteiger on 15.03.22.
//

#ifndef ASL_2022_ASL_UTILS_H
#define ASL_2022_ASL_UTILS_H

#include <asl.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

typedef struct {
    double *M;
    int n_row;
    int n_col;
} Matrix;

typedef struct {
    Matrix V;
    Matrix W;
    Matrix H;
} Matrices;


void allocate_base_matrices(Matrices *matrices, int m, int n, int r);

double rand_from(double min, double max);

void random_matrix_init(Matrix *matrix, double min, double max);

void random_acol_matrix_init(Matrix *V, Matrix *W, int q);

void matrix_allocation(Matrix *matrix, int rows, int cols);

void matrix_deallocation(Matrix *matrix);

void print_matrix(Matrix *matrix);

double norm(Matrix *matrix);

void allocate_from_file(Matrix *matrix, int *r, FILE *file);

#endif //ASL_2022_ASL_UTILS_H
