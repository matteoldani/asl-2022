#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

double rand_from(double min, double max);

void output_matrix(int rows, int cols, int r_out, double min, double max);

int main(int argc, char const *argv[]) {
    srand(time(NULL));

    int rows, cols, r_out;
    double max, min;

    // argument parsing
    if (argc < 4 || argc > 6) {

        printf("Inccorent args, please follow this schema:\n");
        printf("\t- First argument is the rows number (mandatory)\n");
        printf("\t- Second argument is the cols number (mandatory)\n");
        printf("\t- Third argument is the output dimension (r) value (mandatory)\n");
        printf("\t- Fourth argument is the max value (default 1.00)\n");
        printf("\t- Fifth argument is the min value (default 0.00)\n");
        return 0;
    }

    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    r_out = atoi(argv[3]);

    if (argc == 5) {
        sscanf(argv[4], "%lf", &max);
        min = 1.00;
    } else if (argc == 6) {
        sscanf(argv[4], "%lf", &max);
        sscanf(argv[5], "%lf", &min);
    } else {
        max = 1.00;
        min = 0.00;
    }

    // check on the goodnes of params
    if ((!(rows > 0)) || (!(cols > 0)) || !(r_out > 0)) {
        printf("Bad params! Matrix size must be > 0\n\n");
        return 0;
    }

    if ((min < 0) || (min > max)) {
        printf("Bad params! Matrix values must be positive and min < max\n\n");
        return 0;
    }

    output_matrix(rows, cols, r_out, min, max);


    return 0;
}

void output_matrix(int rows, int cols, int r_out, double min, double max) {

    // print the matrix sizes
    fprintf(stdout, "%d %d %d\n", r_out, rows, cols);

    // print the matrix 
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            fprintf(stdout, "%lf ", rand_from(min, max));
        }
        fprintf(stdout, "\n");
    }
}


/* generate a random floating point number from min to max */
double rand_from(double min, double max) {


    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);

}

