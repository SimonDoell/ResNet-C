#include <stdlib.h>
#include <stdint.h>
#include "../matrix.h"
#include "../layer.h"

#ifndef DENSE_H
#define DENSE_H

typedef struct {
    Parameter weights;
    Parameter biases;
} DenseContext;

DenseContext* dense_from_context(void* context);
void dense_get_parameters(void* context, Parameters* parameters);
void dense_free(void* context);
void dense_forward(void* context, const Matrix* restrict activation, Matrix* restrict activation_output);
void dense_backward(void* context, const Matrix* restrict gradient, const Matrix* restrict activation, Matrix* restrict gradient_output);

LayerData dense(uint32_t fan_in, uint32_t fan_out);

#endif