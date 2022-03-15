//
// Created by Viktor Gsteiger on 15.03.22.
//

#include "asl_utils.h"

/**
 * @brief generate a random floating point number from min to max
 * @param min   the minumum possible value
 * @param max   the maximum possible value
 * @return      the random value
 */
double rand_from(double min, double max) {

    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

/**
 * @brief translate a 2D matrix into a vector matrix
 * @param matrix    the 2D matrix to be translated
 * @param vmatrix   the vector matrix to be translated into
 */
void translateMatrixToVMatrix(Matrix *matrix, vMatrix *vmatrix) {
    // TODO: Implement
}