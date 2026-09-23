#include "layer.h"

void optimizer_assert(const Optimizer* restrict optimizer) {
    assert(optimizer);
    assert(
        ( optimizer->context &&  optimizer->free &&  optimizer->step) ||
        (!optimizer->context && !optimizer->free && !optimizer->step)
    );
}

void layerdata_assert(const LayerData* restrict layer_data) {
    assert(layer_data);
    assert(layer_data->context);
    assert(layer_data->forward);
    assert(layer_data->backward);
    assert(layer_data->free);
    assert(layer_data->get_params);
}

void optimizer_free(Optimizer* restrict optimizer) {
    optimizer_assert(optimizer);
    
    optimizer->free(optimizer->context);

    optimizer->context = NULL;
    optimizer->free    = NULL;
    optimizer->step    = NULL;
}

void layerdata_free(LayerData* restrict layer_data) {
    layerdata_assert(layer_data);

    layer_data->free(layer_data->context);

    layer_data->context      = NULL;
    layer_data->forward      = NULL;
    layer_data->backward     = NULL;
    layer_data->free         = NULL;
    layer_data->get_params   = NULL;
    layer_data->param_count  = 0;
}

Layer layer_new(LayerData layer_data, OptimizerFactory optimizer_factory) {
    Layer layer;
    layer.activation_input  = mat_matrix(1, 1);
    layer.activation_output = mat_matrix(1, 1);
    layer.gradient_input    = mat_matrix(1, 1);
    layer.gradient_output   = mat_matrix(1, 1);
    layer.layer_data        = layer_data;
    layer.parameters        = (Parameters){0};
    
    layer_data.get_params(layer.layer_data.context, &layer.parameters);

    if (optimizer_factory)
        layer.optimizer = optimizer_factory(layer.parameters);
    else
        layer.optimizer = (Optimizer){
            .context = NULL,
            .free    = NULL,
            .step    = NULL,
        };

    return layer;
}

void layer_assert(Layer* layer) {
    optimizer_assert(&layer->optimizer);
    layerdata_assert(&layer->layer_data);

    assert(layer->parameters.params);
    mat_assert(&layer->activation_input);
    mat_assert(&layer->activation_output);
    mat_assert(&layer->gradient_input);
    mat_assert(&layer->gradient_output);
}

void layer_free(Layer* layer) {
    layer_assert(layer);

    free(layer->parameters.params);
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

void layer_backward(Layer* layer) {
    layer_assert(layer);
    layer->layer_data.backward(layer->layer_data.context, &layer->gradient_input, &layer->activation_input, &layer->gradient_output);
}

void param_assert(Parameter* parameter) {
    assert(parameter);
    assert(parameter->parameter.rows == parameter->gradient.rows);
    assert(parameter->parameter.cols == parameter->gradient.cols);
}

void param_add_gradient(Parameter* parameter, const Matrix* restrict gradient) {
    param_assert(parameter);
    assert(mat_same_shape(&parameter->gradient, gradient));
    mat_add_inplace(&parameter->gradient, gradient);
}

void layer_apply_gradients(Layer* layer, float lr, uint32_t batch_count) {
    layer_assert(layer);
    assert(batch_count > 0);
    assert(lr != 0.0f);

    layer->layer_data.get_params(layer->layer_data.context, &layer->parameters);

    // Divide the gradients by the batch count
    for (uint32_t i = 0; i < layer->parameters.count; ++i)
        mat_mulf_inplace(&layer->parameters.params[i]->gradient, 1.0f / (float)batch_count);

    // Update the weights with the mean gradients and learning rate
    layer->optimizer.step(layer->optimizer.context, layer->parameters, lr);

    // Zero gradients
    for (uint32_t i = 0; i < layer->parameters.count; ++i)
        mat_zero(&layer->parameters.params[i]->gradient);
}