#include "sgd.h"

void SGD_step(void* context, Parameters parameters, float lr) {
    assert(!context);

    for (uint32_t i = 0; i < parameters.count; ++i) {
        Parameter* parameter = parameters.params[i];
        
        // Update gradients
        mat_mulf_inplace(&parameter->gradient, lr);
        mat_sub_inplace(&parameter->parameter, &parameter->gradient);
        
        // Zero gradients
        mat_zero(&parameter->gradient);
    }
}

Optimizer SGD(const Parameters parameters) {
    return (Optimizer){
        .context = NULL,
        .free    = NULL,
        .step    = SGD_step,
    };
}