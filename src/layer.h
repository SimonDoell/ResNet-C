#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"

#ifndef LAYER_H
#define LAYER_H

typedef struct {
    Matrix* parameter;
    Matrix* gradient;
} Parameter;

typedef struct {
    Parameter* params;
    uint32_t count;
} Parameters;

typedef void(*Free)           (void* context);
typedef void(*Forward)        (void* context, const Matrix* restrict activation, Matrix* restrict activation_output);
typedef void(*Backward)       (void* context, const Matrix* restrict gradient, const Matrix* restrict activation, Matrix* restrict gradient_output);
typedef void(*GetParameters)  (void* context, Parameters* out_parameters);
typedef void(*OptimizerFree)  (void* context_optimizer);
typedef void(*OptimizerStep)  (void* context_optimizer, const Parameter* restrict parameters, float lr);

typedef struct {
    void*          context;
    Free           free;
    Forward        forward;
    Backward       backward;
    GetParameters  get_params;

    void*          context_optimizer;
    OptimizerFree  optimizer_free;
    OptimizerStep  optimizer_step;
    
    Matrix         activation_input;
    Matrix         activation_output;
    Matrix         gradient_input;
    Matrix         gradient_output;
} Layer;

#endif