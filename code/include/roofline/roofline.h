#include <asl.h>

#include <optimizations/optimizations_utils.h>
#include <baselines/baseline1.h>
#include <baselines/baseline2.h>
#include <optimizations/alg_opt_1.h>
#include <optimizations/alg_opt_2.h>
#include <optimizations/optimizations_0.h>
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
#include <optimizations/optimizations_35.h>
#include <optimizations/optimizations_36.h>
#include <optimizations/optimizations_41.h>
#include <optimizations/optimizations_42.h>
#include <optimizations/optimizations_43.h>
#include <optimizations/optimizations_44.h>
#include <optimizations/optimizations_45.h>
#include <optimizations/optimizations_46.h>
#include <optimizations/optimizations_37.h>
#include <optimizations/optimizations_47.h>
#include <optimizations/optimizations_51.h>
#include <optimizations/optimizations_53.h>
#include <optimizations/optimizations_54.h>
#include <optimizations/optimizations_48.h>
#include <optimizations/optimizations_60.h>
#include <optimizations/optimizations_61.h>

typedef double (*fact_function) (Matrix *, Matrix *, Matrix *, int, double);
typedef double (*opt_fact_function)(double *, double*, double*, int, int, int, int, double);

void baseline(int numTests, int input_size, int b,   FILE * fout, fact_function fact_function);
void optimization(int numTests, int input_size, int opt, FILE * fout, opt_fact_function fact_function);
