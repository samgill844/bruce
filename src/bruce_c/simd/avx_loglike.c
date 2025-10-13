#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE 10000000

inline double bruce_loglike(const double y, const double yerr, const double model,
                const double jitter, const int offset)
{
    double wt = 1. / (yerr*yerr + jitter*jitter);
    double loglikeliehood = -0.5*((y - model)*(y - model)*wt);
    // if (offset==1) loglikeliehood += 0.5*log(wt);
    return loglikeliehood;
}

double bruce_loglike_call(const double * y, const double * yerr, const double * model,
                const double jitter, const int offset, const int size)
{
    double sum=0;
    //#pragma omp parallel for simd reduction(+:sum)
    for (int i=0; i< size; i++)
    {
        if (i==0) continue;
        // sum += -0.5*((y[i] - model[i])*(y[i] - model[i])*(1. / (yerr[i]*yerr[i] + jitter*jitter)));
        sum += bruce_loglike(y[i], yerr[i], model[i], jitter, offset);
    }
    return sum;
}

int main(void)
{
    // Allocate arrays
    double *y = (double*) malloc(SIZE * sizeof(double));
    double *yerr = (double*) malloc(SIZE * sizeof(double));
    double *model = (double*) malloc(SIZE * sizeof(double));
    double jitter = 0;
    int offset = 0;

    if (!y || !yerr || !model) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize arrays
    for (int i = 0; i < SIZE; i++) {
        y[i] = 1.0;
        yerr[i] = 0.01;
        model[i] = 1.001;  // you can choose this as well, same as y for zero diff
    }

    
    double result[1000];
    #pragma omp parallel for
    for (int i=0; i<1000; i++)
    {
        result[i] = bruce_loglike_call(y, yerr, model, jitter, offset, SIZE);
    }
    printf("Sum log-likelihood: %.10f\n", result[0]);

    free(y);
    free(yerr);
    free(model);

    return 0;
}
