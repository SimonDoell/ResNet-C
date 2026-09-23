#include "layer.h"

const Optimizer optimizer_NULL = (Optimizer){
    .context = NULL,
    .free    = NULL,
    .step    = NULL,
};

void optimizer_assert(const Optimizer* optimizer) {
    assert(optimizer);
    assert(
        ( optimizer->context &&  optimizer->free &&  optimizer->step) ||
        (!optimizer->context && !optimizer->free && !optimizer->step)
    );
}

void layerdata_assert(const LayerData* layerdata) {
    assert(layerdata);
    assert(layerdata->context);
    assert(layerdata->forward);
    assert(layerdata->free);
    assert(layerdata->get_params);
}

void optimizer_free(Optimizer* optimizer) {
    optimizer_assert(optimizer);
    
    optimizer->free(optimizer->context);

    optimizer->context = NULL;
    optimizer->free    = NULL;
    optimizer->step    = NULL;
}

void layerdata_free(LayerData* layerdata) {
    layerdata_assert(layerdata);

    layerdata->free(layerdata->context);

    layerdata->context    = NULL;
    layerdata->forward    = NULL;
    layerdata->backward   = NULL;
    layerdata->free       = NULL;
    layerdata->get_params = NULL;
}

Layer layer_new(LayerData layer_data, Optimizer optimizer) {
    layerdata_assert(&layer_data);
    optimizer_assert(&optimizer);
    
    return (Layer){
        .layer_data         = layer_data,
        .optimizer          = optimizer,
        .activation_input   = mat_new_tensor(1, 1, 1),
        .activation_output  = mat_new_tensor(1, 1, 1),
        .gradient_input     = mat_new_tensor(1, 1, 1),
        .gradient_output    = mat_new_tensor(1, 1, 1),
    };
}

void layer_assert(const Layer* layer) {
    assert(layer);
    optimizer_assert(&layer->optimizer);
    layerdata_assert(&layer->layer_data);

    mat_assert(&layer->activation_input);
    mat_assert(&layer->activation_output);
    mat_assert(&layer->gradient_input);
    mat_assert(&layer->gradient_output);
}

void layer_free(Layer* layer) {
    layer_assert(layer);

    optimizer_free(&layer->optimizer);
    layerdata_free(&layer->layer_data);

    mat_free(&layer->activation_input);
    mat_free(&layer->activation_output);
    mat_free(&layer->gradient_input);
    mat_free(&layer->gradient_output);
}

void layer_forward(Layer* layer) {
    layer_assert(layer);

    layer->layer_data.forward(layer->layer_data.context, &layer->activation_input, &layer->activation_output);
}

void layer_backward(Layer* layer, float lr) {
    layer_assert(layer);

    layer->layer_data.backward(layer->layer_data.context, &layer->gradient_input, &layer->activation_input, &layer->gradient_output);

    if (layer->optimizer.context == NULL) return;

    Parameters params = {0};

    layer->layer_data.get_params(layer->layer_data.context, &params);
    
    layer->optimizer.step(layer->optimizer.context, &params, lr);
}