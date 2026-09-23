#include <stdio.h>
#include "matrix.h"
#include "layer.h"

int main() {
    // Matrix test
    Matrix res = mat_matrix(1, 1);

    Matrix m1 = mat_matrix(2, 2);
    m1.values[0] = 1.0f;
    m1.values[1] = 2.0f;
    m1.values[2] = 3.0f;
    m1.values[3] = 4.0f;
    
    Matrix m2 = mat_matrix(2, 2);
    m2.values[0] = 5.0f;
    m2.values[1] = 6.0f;
    m2.values[2] = 7.0f;
    m2.values[3] = 8.0f;

    mat_mul_into(&res, &m2, &m1);

    for (uint32_t i = 0; i < res.rows * res.cols; ++i)
        printf("%f ", res.values[i]);
    printf("\n");

    mat_free(&m1);
    mat_free(&m2);
    mat_free(&res);


    // Layer test






    
    return 0;
}