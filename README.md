# Avanced Systenm Lab - 2022

This repo contains the code for the Advanced System Lab course held @ ETHZ

## Goal

Non-Negative Matrix Factorization

## Team

- Daniele Coppola dcoppola@student.ethz.ch
- Viktor Gsteiger vgsteiger@student.ethz.ch
- Masa Nesic mnesic@student.ethz.ch
- Matteo Oldani moldani@student.ethz.ch

## How to use

baseline1.c contains the code for the first baseline implementation. This implementation initializes the factorisation
matriced with random values between 0 and 1.

random_matrix_generator.c is used to craft random populated matrices to be factorized. The program will output to stdout
the output_dimension followed by the matrix dimension and the matrix itsef. This can be used as input for the baseline
binary (./matrix_random_generation | ./baseline1)

## For CBLAS:

Download: http://www.openblas.net/
How to configure: https://charlesjiangxm.wordpress.com/2017/08/03/use-eigen-in-clion/
Documentation: https://developer.apple.com/documentation/accelerate/1513282-cblas_dgemm
Build, install and compile: https://github.com/bgeneto/build-install-compile-openblas

### How to compile the verification:

#### Windows:
gcc -I/opt/openblas/include -pthread -O3 -Wall baseline2/baseline2.c baseline1/baseline1.c asl_utils/asl_utils.c verification/verify.c -o verification.out -L/opt/openblas/lib -lm -lpthread -lopenblas

#### MacOS:
gcc -O3 baseline2/baseline2.c baseline1/baseline1.c asl_utils/asl_utils.c verification/verify.c -framework Accelerate -lm -o verification.out
