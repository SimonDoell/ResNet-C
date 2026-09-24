#include <stdio.h>
#include <time.h>
#include "matrix.h"
#include "layer.h"
#include "layers/dense.h"
#include "optimizers/sgd.h"

int main() {
    srand(time(0));
    
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
    Layer layer1 = layer_new(dense(1, 16), SGD);
    Layer layer2 = layer_new(dense(16, 1), SGD);

    Matrix input = mat_vector(1);
    input.values[0] = 1.0f;

    mat_copy_into(&layer1.activation_input, &input);
    layer_forward(&layer1);

    mat_copy_into(&layer2.activation_input, &layer1.activation_output);
    layer_forward(&layer2);

    Matrix output = mat_vector(1);
    mat_copy_into(&output, &layer2.activation_output);

    printf("Output %ix%i: %f\n", output.rows, output.cols, output.values[0]);
    
    layer_free(&layer1);
    layer_free(&layer2);
    mat_free(&input);
    mat_free(&output);
    
    return 0;
}