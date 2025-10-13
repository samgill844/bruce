#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define SIZE 2048
#define NUM_POINTS 100000

double floor_sqrt(double x) {
    return (x > 0) ? sqrt(x) : 0.0;
}

double area_triangle(double x1, double y1, double x2, double y2, double x3, double y3) {
    return 0.5 * fabs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
}

double distance(double x1, double y1, double x2, double y2) {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

double area_arc(double x1, double y1, double x2, double y2, double r) {
    double a = distance(x1, y1, x2, y2);
    double theta = 2.0 * asin(0.5 * a / r);
    return 0.5 * r * r * (theta - sin(theta));
}

double circular_overlap_with_box_core(double xmin, double ymin, double xmax, double ymax, double r) {
    double area = 0.0;
    
    if (xmin * xmin + ymin * ymin > r * r) {
        area = 0.0; // Box is completely outside the circle
    } else if (xmax * xmax + ymax * ymax < r * r) {
        area = (xmax - xmin) * (ymax - ymin); // Box is completely inside the circle
    } else {
        double d1 = floor_sqrt(xmax * xmax + ymin * ymin);
        double d2 = floor_sqrt(xmin * xmin + ymax * ymax);
        
        if (d1 < r && d2 < r) {
            double x1 = floor_sqrt(r * r - ymax * ymax), y1 = ymax;
            double x2 = xmax, y2 = floor_sqrt(r * r - xmax * xmax);
            area = ((xmax - xmin) * (ymax - ymin) -
                    area_triangle(x1, y1, x2, y2, xmax, ymax) +
                    area_arc(x1, y1, x2, y2, r));
        } else if (d1 < r) {
            double x1 = xmin, y1 = floor_sqrt(r * r - xmin * xmin);
            double x2 = xmax, y2 = floor_sqrt(r * r - xmax * xmax);
            area = (area_arc(x1, y1, x2, y2, r) +
                    area_triangle(x1, y1, x1, ymin, xmax, ymin) +
                    area_triangle(x1, y1, x2, ymin, x2, y2));
        } else if (d2 < r) {
            double x1 = floor_sqrt(r * r - ymin * ymin), y1 = ymin;
            double x2 = floor_sqrt(r * r - ymax * ymax), y2 = ymax;
            area = (area_arc(x1, y1, x2, y2, r) +
                    area_triangle(x1, y1, xmin, y1, xmin, ymax) +
                    area_triangle(x1, y1, xmin, y2, x2, y2));
        } else {
            double x1 = floor_sqrt(r * r - ymin * ymin), y1 = ymin;
            double x2 = xmin, y2 = floor_sqrt(r * r - xmin * xmin);
            area = (area_arc(x1, y1, x2, y2, r) +
                    area_triangle(x1, y1, x2, y2, xmin, ymin));
        }
    }
    return area;
}


double circular_overlap_with_box(double xmin, double ymin, double xmax, double ymax, double R) {
    // Fully in one quadrant
    if (xmin >= 0 && xmax >= 0 && ymin >= 0 && ymax >= 0)
        return circular_overlap_with_box_core(xmin, ymin, xmax, ymax, R); // Q1
    if (xmin >= 0 && xmax >= 0 && ymin < 0 && ymax < 0)
        return circular_overlap_with_box_core(xmin, -ymax, xmax, -ymin, R); // Q2
    if (xmin < 0 && xmax < 0 && ymin < 0 && ymax < 0)
        return circular_overlap_with_box_core(-xmax, -ymax, -xmin, -ymin, R); // Q3
    if (xmin < 0 && xmax < 0 && ymin >= 0 && ymax >= 0)
        return circular_overlap_with_box_core(-xmax, ymin, -xmin, ymax, R); // Q4

    // Spanning two quadrants
    if (xmin >= 0 && xmax >= 0 && ymin < 0 && ymax >= 0) // Q1 and Q2
        return circular_overlap_with_box_core(xmin, 0, xmax, ymax, R) +
               circular_overlap_with_box_core(xmin, 0, xmax, -ymin, R);

    if (xmin < 0 && xmax >= 0 && ymin < 0 && ymax < 0) // Q2 and Q3
        return circular_overlap_with_box_core(0, -ymax, xmax, -ymin, R) +
               circular_overlap_with_box_core(0, -ymax, -xmin, -ymin, R);

    if (xmin < 0 && xmax < 0 && ymin < 0 && ymax >= 0) // Q3 and Q4
        return circular_overlap_with_box_core(-xmax, 0, -xmin, ymax, R) +
               circular_overlap_with_box_core(-xmax, 0, -xmin, -ymin, R);

    if (xmin < 0 && xmax >= 0 && ymin >= 0 && ymax >= 0) // Q4 and Q1
        return circular_overlap_with_box_core(0, ymin, -xmin, ymax, R) +
               circular_overlap_with_box_core(0, ymin, xmax, ymax, R);

    // Box covering the origin
    if (xmin < 0 && xmax >= 0 && ymin < 0 && ymax >= 0)
        return circular_overlap_with_box_core(0, 0, xmax, ymax, R) +
               circular_overlap_with_box_core(0, 0, xmax, -ymin, R) +
               circular_overlap_with_box_core(0, 0, -xmin, -ymin, R) +
               circular_overlap_with_box_core(0, 0, -xmin, ymax, R);

    return 0.0;
}



double sum_circle(double **data, int width, int height, double X, double Y, double R) {
    // Compute the extent of the integration box
    int min_extentx = (int)floor(X - R + 0.5);
    int max_extentx = (int)ceil(X + R - 0.5);
    int min_extenty = (int)floor(Y - R + 0.5);
    int max_extenty = (int)ceil(Y + R - 0.5);

    // Clip to image boundaries
    if (min_extentx < 0) min_extentx = 0;
    if (max_extentx >= width) max_extentx = width - 1;
    if (min_extenty < 0) min_extenty = 0;
    if (max_extenty >= height) max_extenty = height - 1;

    // Compute flux
    double flux = 0.0;
    for (int j = min_extentx; j <= max_extentx; j++) { // Cycle columns (x axis)
        for (int i = min_extenty; i <= max_extenty; i++) { // Cycle rows (y axis)
            double xmin = j - 0.5 - X;
            double xmax = j + 0.5 - X;
            double ymin = i - 0.5 - Y;
            double ymax = i + 0.5 - Y;
            double frac = circular_overlap_with_box(xmin, ymin, xmax, ymax, R);
            flux += data[i][j] * frac;
        }
    }
    return flux;
}

int main() {
    srand(time(NULL));

    // Allocate image
    double **image = (double **)malloc(SIZE * sizeof(double *));
    for (int i = 0; i < SIZE; i++) {
        image[i] = (double *)malloc(SIZE * sizeof(double));
        for (int j = 0; j < SIZE; j++) {
            image[i][j] = (double)rand() / RAND_MAX;
        }
    }

    // Allocate photometry arrays dynamically
    double *x_values = (double *)malloc(NUM_POINTS * sizeof(double));
    double *y_values = (double *)malloc(NUM_POINTS * sizeof(double));
    double *results = (double *)malloc(NUM_POINTS * sizeof(double));

    // Parallel initialization of random coordinates
    for (int j=0; j<20000; j++){
    #pragma omp parallel
    {
        unsigned int seed = time(NULL) + omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < NUM_POINTS; i++) {
            x_values[i] = (double)(rand_r(&seed) % SIZE);
            y_values[i] = (double)(rand_r(&seed) % SIZE);
        }
    }}

//     // Parallel photometry calculation
//     #pragma omp parallel for
//     for (int j=0; j<200; j++){
//         for (int i = 0; i < NUM_POINTS; i++) {
//             results[i] = sum_circle(image, SIZE,SIZE,x_values[i], y_values[i], 3.0);
//         }
// }

    printf("First photometry result: %f\n", results[0]);

    // Free allocated memory
    for (int i = 0; i < SIZE; i++) {
        free(image[i]);
    }
    free(image);
    free(x_values);
    free(y_values);
    free(results);

    return 0;
}
