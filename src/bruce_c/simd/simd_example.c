// simd_example.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void scale_array(double *restrict a, double scale, int n) {
    #pragma omp parallel for simd
    for (int i = 0; i < n; i++) {
        a[i] *= scale;
    }
}

int main() {
    long n = 10000000000;
    double *a = aligned_alloc(32, n * sizeof(double)); // 32-byte aligned for AVX

    // Initialize array
    for (int i = 0; i < n; i++) {
        a[i] = i * 0.5;
    }

    double scale = 2.0;
    scale_array(a, scale, n);

    // Print a few values to confirm
    for (int i = 0; i < 5; i++) {
        printf("a[%d] = %f\n", i, a[i]);
    }

    free(a);
    return 0;
}
