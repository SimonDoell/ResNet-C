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

void dense_free(void* context) {
    DenseContext* dense = dense_from_context(context);

    param_free(&dense->biases);
    param_free(&dense->weights);
}

void dense_forward(void* context, const Matrix* restrict activation, Matrix* restrict activation_output) {
    DenseContext* dense = dense_from_context(context);

    mat_mul_into   (activation_output, &dense->weights.parameter, activation);
    mat_add_inplace(activation_output, &dense->biases .parameter);
}

void dense_backward(void* context, const Matrix* restrict gradient, const Matrix* restrict activation, Matrix* restrict gradient_output) {
    DenseContext* dense = dense_from_context(context);

    uint32_t output_size = dense->weights.parameter.rows;
    uint32_t input_size  = dense->weights.parameter.cols;

    mat_mul_transposed_into(gradient_output, &dense->weights.parameter /*implicitely transposed by the function*/, gradient);

    mat_add_inplace(&dense->biases.gradient, gradient);

    mat_mul_accumulate_into(&dense->weights.gradient, gradient, activation);
}

LayerData dense(uint32_t fan_in, uint32_t fan_out) {
    DenseContext* context = malloc(sizeof(DenseContext));

    context->weights   = param_matrix(fan_out, fan_in);
    context->biases    = param_vector(fan_out);

    return (LayerData){
        .context      = context,
        .free         = dense_free,
        .forward      = dense_forward,
        .backward     = dense_backward,
        .param_count  = 2,
        .get_params   = dense_get_parameters,
    };
}