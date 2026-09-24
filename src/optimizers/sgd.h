#include <stdlib.h>
#include <stdint.h>
#include "../matrix.h"
#include "../layer.h"

#ifndef SGD_H
#define SGD_H

// typedef struct {
//     void*           context;
//     Free            free;
//     OptimizerStep   step;
// } Optimizer;

// typedef void(*Free)          (void* context);
// typedef void(*OptimizerStep) (void* context, Parameters parameters, float lr);
// typedef Optimizer(*OptimizerFactory)(const Parameters parameters);

void SGD_step(void* context, Parameters parameters, float lr);
Optimizer SGD(const Parameters parameters);

#endif