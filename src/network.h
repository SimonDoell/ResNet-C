#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"
#include "layer.h"

#ifndef NETWORK_H
#define NETWORK_H

typedef enum {
    NODE_ADD, NODE_CONCAT
} NodeOp;

typedef struct {
    Layer       layer;
    NodeOp      node_op;
    uint32_t*   input;
    uint32_t*   output;
    uint32_t    input_count;
    uint32_t    output_count;
} Node;

typedef struct {
    float       learning_rate;
    Node*       nodes;
    uint32_t*   exec_order;
    uint32_t    node_count;
} Network;

Node node_init(Layer layer, NodeOp node_op);
void node_free(Node* restrict n);
void node_assert(Node* restrict n);
void node_connect(Node* restrict n1, uint32_t index_1, Node* restrict n2, uint32_t index_2);

Network network_init();
void network_free(Network* restrict network);
uint32_t network_add_layer(Network* restrict network, Layer layer);
void network_change_layer_op(Network* restrict network, uint32_t layer_idx, NodeOp node_op);
void network_set_exec_order(Network* restrict network);

void network_forward(Network* restrict network, const Matrix* restrict activation);
void network_backward(Network* restrict network, const Matrix* restrict desired_output);
void network_apply_gradients(Network* restrict network);

#endif