#include "network.h"

Node node_init(Layer layer, NodeOp node_op) {
    layer_assert(&layer);
    
    return (Node){
        .layer         = layer,
        .node_op       = node_op,
        .input         = NULL,
        .output        = NULL,
        .input_count   = 0,
        .output_count  = 0,
    };
}

void node_free(Node* restrict n) {
    node_assert(n);

    layer_free(&n->layer);
    if (n->input)  free(n->input);
    if (n->output) free(n->output);

    n->input  = NULL;
    n->output = NULL;
}

void node_assert(Node* restrict n) {
    assert(n);
    layer_assert(&n->layer);
}

void node_connect(Node* restrict n1, uint32_t index_1, Node* restrict n2, uint32_t index_2) {
    node_assert(n1);
    node_assert(n2);

    n1->output = realloc(n1->output, ++n1->output_count * sizeof(uint32_t));
    n2->input  = realloc(n2->input,  ++n2->input_count  * sizeof(uint32_t));

    n1->output[n1->output_count - 1] = index_2;
    n2->input [n2->input_count  - 1] = index_1;
}

Network network_init() {
    return (Network){
        .nodes      = NULL,
        .exec_order = NULL,
        .node_count = 0,
    };
}

void network_free(Network* restrict network) {
    assert(network);

    for (uint32_t i = 0; i < network->node_count; ++i)
        node_free(&network->nodes[i]);

    if (network->exec_order)
        free(network->exec_order);

    network->exec_order = NULL;
}

uint32_t network_add_layer(Network* restrict network, Layer layer) {
    assert(network);

    network->nodes = realloc(network->nodes, ++network->node_count * sizeof(Node));

    uint32_t index = network->node_count - 1;
    
    network->nodes[index] = node_init(layer, NODE_ADD);

    if (network->node_count > 1)
        node_connect(&network->nodes[index - 1], index - 1, &network->nodes[index], index);

    if (network->exec_order) {
        free(network->exec_order);
        network->exec_order = NULL;
    }
    
    return index;
}

void network_change_layer_op(Network* restrict network, uint32_t layer_idx, NodeOp node_op) {
    assert(network);
    assert(layer_idx < network->node_count);

    network->nodes[layer_idx].node_op = node_op;
}

void network_set_exec_order(Network* restrict network) {
    assert(network);

    network->exec_order = realloc(network->exec_order, network->node_count * sizeof(uint32_t));

    for (uint32_t i = 0; i < network->node_count; ++i) {
        network->exec_order[i] = i;
    }
}

void network_forward(Network* restrict network, const Matrix* restrict activation) {
    assert(network);
    assert(network->node_count >= 1);
    assert(network->nodes[network->exec_order[0]].node_op == NODE_ADD);

    Layer* layer0 = &network->nodes[network->exec_order[0]].layer;

    mat_copy_into(&layer0->activation_input, activation);
    layer_forward(layer0);
    
    for (uint32_t i = 1; i < network->node_count; ++i) {
        Node* nodei = &network->nodes[i];

        if (nodei->node_op == NODE_ADD) {
            mat_zero(&nodei->layer.activation_input);

            for (uint32_t in = 0; in < nodei->input_count; ++in)
                mat_add_inplace(&nodei->layer.activation_input, &network->nodes[nodei->input[in]].layer.activation_output);

            layer_forward(&nodei->layer);
            
        } else if (nodei->node_op == NODE_CONCAT) {
            assert(false);
        }
        else assert(false);
    }
}

void network_backward(Network* restrict network, const Matrix* restrict desired_output) {
    assert(network);
    assert(network->node_count >= 1);
}

void network_apply_gradients(Network* restrict network) {
    assert(network);
    assert(network->node_count >= 1);
}