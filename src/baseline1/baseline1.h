#ifndef ASL_2022_BASELINE1_H
#define ASL_2022_BASELINE1_H

/* Parameters */
#define EPSILON 0.5
#define MAX_ITERATION  500

typedef struct {
    double **M;
    int n_row;
    int n_col;
} Matrix;

double nnm_factorization_bs1(Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon);

#endif //ASL_2022_BASELINE1_H
