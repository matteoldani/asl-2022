#ifndef ASL_2022_BASELINE2_H
#define ASL_2022_BASELINE2_H

/* Parameters */
#define MAX_ITERATION  500

typedef struct {
    double *M;
    int n_row;
    int n_col;
} vMatrix;

double nnm_factorization_bs2(vMatrix *V, vMatrix *W, vMatrix *H, int maxIteration, double epsilon);

#endif //ASL_2022_BASELINE2_H
