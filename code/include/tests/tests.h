#include <asl.h>
#include <math.h>
#include <baselines/baseline1.h>
#include <baselines/baseline2.h>
#include <baselines/baselines_utils.h>
#include <optimizations/optimizations_baseline.h>
#include<unistd.h> 

#ifndef WIN32
#include <sys/time.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifdef __x86_64__
#include <performance/tsc_x86.h>
#endif
