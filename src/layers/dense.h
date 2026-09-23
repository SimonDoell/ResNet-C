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

#endif