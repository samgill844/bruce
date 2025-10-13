// simd_avx_omp.c
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>

#define SIZE (1024 * 1024 * 64)  // 64 million doubles

double time_diff_sec(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
}

int main() {
    #pragma omp parallel
    {
        #pragma omp single
        printf("OpenMP threads = %d\n", omp_get_num_threads());
    }

    double* a = aligned_alloc(32, SIZE * sizeof(double));
    double* b = aligned_alloc(32, SIZE * sizeof(double));
    double* result = aligned_alloc(32, SIZE * sizeof(double));

    if (!a || !b || !result) {
        perror("alloc");
        return 1;
    }

    for (int i = 0; i < SIZE; ++i) {
        a[i] = 1.234;
        b[i] = 5.678;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int j=0;j<100;j++){
    #pragma omp parallel for
    for (int i = 0; i < SIZE; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_mul_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }}

    clock_gettime(CLOCK_MONOTONIC, &end);

    printf("result[0] = %f\n", result[0]);
    printf("Time: %.6f seconds\n", time_diff_sec(start, end));

    free(a);
    free(b);
    free(result);
    return 0;
}
