//
// Created by Viktor Gsteiger on 15.03.22.
//

#ifndef ASL_2022_ASL_UTILS_H
#define ASL_2022_ASL_UTILS_H

typedef struct {
    double **M;
    int n_row;
    int n_col;
} Matrix;

typedef struct {
    double *M;
    int n_row;
    int n_col;
} vMatrix;

typedef struct {
    Matrix V;
    Matrix W;
    Matrix H;
} Baseline1Matrices;

typedef struct {
    vMatrix V;
    vMatrix W;
    vMatrix H;
} Baseline2Matrices;

void initialise_bs1_matrices(Baseline1Matrices *bs1, int m, int n, int r);

void initialise_bs2_matrices(Baseline2Matrices *bs2, int m, int n, int r);

void translate_matrix_to_v_matrix(Matrix *matrix, vMatrix *vmatrix);

double rand_from(double min, double max);

void allocate_matrix_v_matrix(Matrix *matrix, vMatrix *vmatrix);

void generate_random_matrices(Matrix *matrix, vMatrix *vmatrix, double min, double max);

void random_v_matrix_init(vMatrix *matrix, double min, double max);

void v_matrix_allocation(vMatrix *matrix);

void v_matrix_deallocation(vMatrix *matrix);

void random_matrix_init(Matrix *matrix, double min, double max);

void matrix_allocation(Matrix *matrix);

void matrix_deallocation(Matrix *matrix);

#endif //ASL_2022_ASL_UTILS_H
