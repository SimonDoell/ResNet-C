#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t batches;
    float* values;
} Matrix;

Matrix mat_new_matrix(uint32_t rows, uint32_t cols);
Matrix mat_new_vector(uint32_t rows);
Matrix mat_new_tensor(uint32_t rows, uint32_t cols, uint32_t batches);
void mat_free(Matrix* restrict m);

Matrix mat_copy(const Matrix* restrict m);
void mat_move_into(Matrix* restrict dest, Matrix* restrict src);
void mat_resize(Matrix* restrict m, uint32_t rows, uint32_t cols, uint32_t batches);

void mat_assert(const Matrix* restrict m);
bool mat_same_shape(const Matrix* restrict m1, const Matrix* restrict m2);

uint32_t mat_index(uint32_t r, uint32_t c, uint32_t b, uint32_t rows, uint32_t cols);
float mat_at_const(const Matrix* restrict m, uint32_t r, uint32_t c, uint32_t b);
float* mat_at(Matrix* restrict m, uint32_t r, uint32_t c, uint32_t b);

void mat_add_inplace(Matrix* restrict m1, const Matrix* restrict m2);
void mat_sub_inplace(Matrix* restrict m1, const Matrix* restrict m2);
Matrix mat_transposed(Matrix* restrict m);

Matrix mat_mul(const Matrix* restrict m1, const Matrix* restrict m2);

void mat_add_copy_to_batch(Matrix* restrict m1, const Matrix* restrict m2);

#endif