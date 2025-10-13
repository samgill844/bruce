// simd_align_test.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>  // for AVX intrinsics

#define SIZE (1024 * 1024 * 64)  // 64M doubles = 512MB
#define ALIGNMENT 32             // For AVX

double scale_array(double* a, size_t n, double scale) {
    clock_t start = clock();
    for (size_t i = 0; i < n; i++) {
        a[i] *= scale;
    }
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

int main() {
    double* a_malloc = malloc(SIZE * sizeof(double));
    double* a_aligned = aligned_alloc(ALIGNMENT, SIZE * sizeof(double));

    if (!a_malloc || !a_aligned) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    // Initialize both arrays with dummy values
    for (size_t i = 0; i < SIZE; i++) {
        a_malloc[i] = 1.0;
        a_aligned[i] = 1.0;
    }

    double time_malloc = scale_array(a_malloc, SIZE, 1.00001);
    double time_aligned = scale_array(a_aligned, SIZE, 1.00001);

    printf("Time with malloc:        %.4f seconds\n", time_malloc);
    printf("Time with aligned_alloc: %.4f seconds\n", time_aligned);

    free(a_malloc);
    free(a_aligned);

    return 0;
}
