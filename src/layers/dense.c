#include "dense.h"

DenseContext* dense_from_context(void* context) {
    assert(context);

    DenseContext* dense = (DenseContext*)context;

    param_assert(&dense->biases);
    param_assert(&dense->weights);

    return dense;
}

void dense_get_parameters(void* context, Parameters* parameters) {
    DenseContext* dense = dense_from_context(context);

    assert(parameters->count == 2);

    parameters->params[0] = &dense->biases;
    parameters->params[1] = &dense->weights;
}
