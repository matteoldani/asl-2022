#include <asl.h>

#include <optimizations/optimizations_utils.h>
#include <baselines/baseline1.h>
#include <baselines/baseline2.h>
#include <optimizations/alg_opt_1.h>
#include <optimizations/alg_opt_2.h>
#include <optimizations/optimizations_1.h>
#include <optimizations/optimizations_2.h>
#include <optimizations/optimizations_3.h>
#include <optimizations/optimizations_21.h>
#include <optimizations/optimizations_22.h>
#include <optimizations/optimizations_23.h>
#include <optimizations/optimizations_24.h>
#include <optimizations/optimizations_31.h>
#include <optimizations/optimizations_32.h>
#include <optimizations/optimizations_33.h>
#include <optimizations/optimizations_34.h>


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
typedef double (*opt_fact_function)(double *, double*, double*, int, int, int, int, double);
typedef myInt64 (*opt_fact_cost)(int, int, int, int, int, int, int);


void     baseline(int numTests, int min, int max, int b,   FILE * fout, fact_function fact_function, fact_cost fact_cost_function);
void optimization(int numTests, int min, int max, int opt, FILE * fout, opt_fact_function fact_function, opt_fact_cost fact_cost_function);
