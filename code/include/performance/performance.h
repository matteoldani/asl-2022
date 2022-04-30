#include <asl.h>

#include <baselines/baseline1.h>
#include <baselines/baseline2.h>


#ifndef WIN32
#include <sys/time.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifdef __x86_64__
#include <performance/tsc_x86.h>
#endif

#define NUM_RUNS 10
#define CYCLES_REQUIRED 1e8
#define FREQUENCY 3.5e9	
#define CALIBRATE

typedef myInt64 (*fact_cost) (Matrix *, Matrix *, Matrix *, int);
typedef double (*fact_function) (Matrix *, Matrix *, Matrix *, int, double);


void baseline(int numTests, int min, int max, int b, FILE * fout, fact_function fact_function, fact_cost fact_cost_function);