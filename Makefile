# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects

.PHONY = all clean b1 generator

CC = gcc                        # compiler to use

SRCS := $(wildcard *.c)
BINS := $(SRCS:%.c=%)

all: ${BINS}

b1:
	@echo "Compiling baseline 1.."
	${CC} -o baseline1 baseline1.c

generator:
	@echo "Compiling matrix generator.."
	${CC} -o random_matrix_generator random_matrix_generator.c



clean:
	@echo "Cleaning up..."
	rm -rvf *.o ${BINS}