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

    free(context);
}

void dense_forward(void* context, const Matrix* restrict activation, Matrix* restrict activation_output) {
    DenseContext* dense = dense_from_context(context);

    mat_mul_into   (activation_output, &dense->weights.parameter, activation);
    mat_add_inplace(activation_output, &dense->biases .parameter);
}

void dense_backward(void* context, const Matrix* restrict gradient, const Matrix* restrict activation, Matrix* restrict gradient_output) {
    DenseContext* dense = dense_from_context(context);

    mat_mul_transposed_m1_into(gradient_output, &dense->weights.parameter /*implicitely transposed by the function*/, gradient);

    mat_add_inplace(&dense->biases.gradient, gradient);

    mat_mul_accumulate_transposed_m2_into(&dense->weights.gradient, gradient, activation /*implicitely transposed by the function*/);
}

LayerData dense(uint32_t fan_in, uint32_t fan_out) {
    DenseContext* context = malloc(sizeof(DenseContext));

    context->weights   = param_matrix(fan_out, fan_in);
    context->biases    = param_vector(fan_out);

    // Xavier init
    float range = sqrtf(6.0f / (float)(fan_in + fan_out));
    
    for (uint32_t i = 0; i < context->weights.parameter.rows * context->weights.parameter.cols; ++i) {
        context->weights.parameter.values[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * range;
    }

    return (LayerData){
        .context      = context,
        .free         = dense_free,
        .forward      = dense_forward,
        .backward     = dense_backward,
        .param_count  = 2,
        .get_params   = dense_get_parameters,
    };
}