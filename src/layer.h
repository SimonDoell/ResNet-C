#include <stdlib.h>
#include <stdint.h>
#include "matrix.h"

#ifndef LAYER_H
#define LAYER_H

typedef struct {
    Matrix       parameter;
    Matrix       gradient;
} Parameter;

typedef struct {
    Parameter**  params; // Array of pointers to parameter(s)
    uint32_t     count;
} Parameters;

typedef void(*Free)          (void* context);
typedef void(*OptimizerStep) (void* context, Parameters parameters, float lr);
typedef void(*Forward)       (void* context, const Matrix* restrict activation, Matrix* restrict activation_output);
typedef void(*Backward)      (void* context, const Matrix* restrict gradient, const Matrix* restrict activation, Matrix* restrict gradient_output);
typedef void(*GetParameters) (void* context, Parameters* parameters);

typedef struct {
    void*           context;
    Free            free;
    OptimizerStep   step;
} Optimizer;

typedef Optimizer(*OptimizerFactory)(const Parameters parameters);

typedef struct {
    void*           context;
    Free            free;
    Forward         forward;
    Backward        backward;
    uint32_t        param_count;
    GetParameters   get_params;
} LayerData;

typedef struct {
    Optimizer       optimizer;
    LayerData       layer_data;
    Parameters      parameters;

    Matrix          activation_input;
    Matrix          activation_output;
    Matrix          gradient_input;
    Matrix          gradient_output;
} Layer;

void param_assert(Parameter* parameter);
void param_add_gradient(Parameter* parameter, const Matrix* restrict gradient);

void optimizer_assert(const Optimizer* restrict optimizer);
void layerdata_assert(const LayerData* restrict layer_data);
void optimizer_free(Optimizer* restrict optimizer);
void layerdata_free(LayerData* restrict layer_data);

Layer layer_new(LayerData layer_data, OptimizerFactory optimizer_factory);
void layer_assert(Layer* layer);
void layer_free(Layer* layer);

void layer_forward(Layer* layer);
void layer_backward(Layer* layer);
void layer_apply_gradients(Layer* layer, float lr, uint32_t batch_count);

#endif