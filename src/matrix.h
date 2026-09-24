#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    uint32_t  rows;
    uint32_t  cols;
    float*    values;
} Matrix;

Matrix mat_matrix(uint32_t rows, uint32_t cols);
Matrix mat_vector(uint32_t rows);

void mat_assert(const Matrix* restrict m);
bool mat_same_shape(const Matrix* restrict m1, const Matrix* restrict m2);

void mat_free(Matrix* restrict m);
void mat_zero(Matrix* restrict m);
void mat_resize(Matrix* restrict m, uint32_t rows, uint32_t cols);

void mat_copy_into(Matrix* restrict dest, const Matrix* restrict src);
void mat_move_into(Matrix* restrict dest, Matrix* restrict src);

uint32_t mat_index(uint32_t row, uint32_t col, uint32_t rows);
float mat_at_const(const Matrix* restrict m, uint32_t row, uint32_t col);
float* mat_at(Matrix* restrict m, uint32_t row, uint32_t col);

void mat_add_inplace(Matrix* restrict m1, const Matrix* restrict m2);
void mat_sub_inplace(Matrix* restrict m1, const Matrix* restrict m2);
void mat_mulf_inplace(Matrix* restrict m, float f);

void mat_transpose_into(Matrix* restrict dest, const Matrix* restrict src);

void mat_mul_into(Matrix* restrict dest, const Matrix* restrict m1, const Matrix* restrict m2);
void mat_mul_accumulate_transposed_m2_into(Matrix* restrict dest, const Matrix* restrict m1, const Matrix* restrict m2);
void mat_mul_transposed_m1_into(Matrix* restrict dest, const Matrix* restrict m1_T, const Matrix* restrict m2);

#endif