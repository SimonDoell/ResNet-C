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
typedef void(*OptimizerFree)  (void* context);
typedef void(*OptimizerStep)  (void* context, const Parameters* restrict parameters, float lr);

typedef struct {
    void*          context;
    OptimizerFree  free;
    OptimizerStep  step;
} Optimizer;

// NULLed optimizer for layers that do not have learnable parameters
extern const Optimizer optimizer_NULL;

typedef struct {
    void*          context;
    Free           free;
    Forward        forward;
    Backward       backward;
    GetParameters  get_params;
} LayerData;


typedef struct {
    LayerData      layer_data;
    Optimizer      optimizer;
    
    Matrix         activation_input;
    Matrix         activation_output;
    Matrix         gradient_input;
    Matrix         gradient_output;
} Layer;

void optimizer_assert(const Optimizer* optimizer);
void layerdata_assert(const LayerData* layerdata);
void optimizer_free(Optimizer* optimizer);
void layerdata_free(LayerData* layerdata);

Layer layer_new(LayerData layer_data, Optimizer optimizer);

void layer_assert(const Layer* layer);
void layer_free(Layer* layer);

void layer_forward(Layer* layer);
void layer_backward(Layer* layer, float lr);

#endif