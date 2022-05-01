#include <asl.h>
#include <math.h>
#include <baselines/baseline1.h>
#include <baselines/baseline2.h>
#include <baselines/baselines_utils.h>
#include<unistd.h> 

int test_matrix_mult(void (*mmul) (Matrix *A, Matrix *B, Matrix *R));
int test_matrix_ltrans_mult(void (*mmulltrans) (Matrix *A, Matrix *B, Matrix *R));
int test_matrix_rtrans_mult(void (*mmulrtrans) (Matrix *A, Matrix *B, Matrix *R));
int test_nnm(double (*nnm) (Matrix *V, Matrix *W, Matrix *H, int maxIteration, double epsilon));
