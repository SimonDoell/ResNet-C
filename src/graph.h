#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "layer.h"

#ifndef GRAPH_H
#define GRAPH_H

typedef enum : uint8_t {
    NODE_ADD, NODE_CONCAT
} NodeOp;

typedef struct {
    uint32_t    layer;
    NodeOp      op;      // A node can add or cancat its INPUTS !

    uint32_t*   input;
    uint32_t    input_count;
    uint32_t*   output;
    uint32_t    output_count;
} Node;

typedef struct {
    Layer*      layers;
    uint32_t*   exec_order;
    uint32_t    layer_count;
} Network;

#endif