#include "matrix.h"

Matrix mat_matrix(uint32_t rows, uint32_t cols) {
    assert(rows > 0);
    assert(cols > 0);
    
    return (Matrix){
        .rows    = rows,
        .cols    = cols,
        .values  = calloc(sizeof(float), rows * cols),
    };
}

Matrix mat_vector(uint32_t rows) {
    return mat_matrix(rows, 1);
}

void mat_assert(const Matrix* restrict m) {
    assert(m);
    assert(m->rows > 0);
    assert(m->cols > 0);
    assert(m->values);
}

void mat_free(Matrix* restrict m) {
    mat_assert(m);
    
    free(m->values);

    m->cols    = 0;
    m->rows    = 0;
    m->values  = NULL;
}

bool mat_same_shape(const Matrix* restrict m1, const Matrix* restrict m2) {
    mat_assert(m1);
    mat_assert(m2);
    return (m1->rows == m2->rows) && (m1->cols == m2->cols);
}

void mat_zero(Matrix* restrict m) {
    mat_assert(m);
    memset(m->values, 0x00, m->rows * m->cols * sizeof(float));
}

void mat_resize(Matrix* restrict m, uint32_t rows, uint32_t cols) {
    mat_assert(m);

    if (m->rows == rows && m->cols == cols) return;

    uint32_t old_count = m->rows * m->cols;
    uint32_t new_count =    rows *    cols;

    m->values = realloc(m->values, new_count * sizeof(float));
    assert(m->values);

    m->rows = rows;
    m->cols = cols;

    for (uint32_t i = old_count; i < new_count; ++i)
        m->values[i] = 0.0f;
}

void mat_copy_into(Matrix* restrict dest, const Matrix* restrict src) {
    mat_assert(dest);
    mat_assert(src);

    mat_resize(dest, src->rows, src->cols);

    uint32_t count = src->rows * src->cols;

    for (uint32_t i = 0; i < count; ++i)
        dest->values[i] = src->values[i];
}

void mat_move_into(Matrix* restrict dest, Matrix* restrict src) {
    mat_assert(dest);
    mat_assert(src);

    mat_free(dest);

    dest->rows   = src->rows;
    dest->cols   = src->cols;
    dest->values = src->values;

    src->rows    = 0;
    src->cols    = 0;
    src->values  = NULL;
}

uint32_t mat_index(uint32_t row, uint32_t col, uint32_t rows) {
    return (row) + (col * rows); // Column-major!
}

float mat_at_const(const Matrix* restrict m, uint32_t row, uint32_t col) {
    mat_assert(m);
    assert(row < m->rows);
    assert(col < m->cols);
    return m->values[mat_index(row, col, m->rows)];
}

float* mat_at(Matrix* restrict m, uint32_t row, uint32_t col) {
    mat_assert(m);
    assert(row < m->rows);
    assert(col < m->cols);
    return &m->values[mat_index(row, col, m->rows)];
}

void mat_add_inplace(Matrix* restrict m1, const Matrix* restrict m2) {
    mat_assert(m1);
    mat_assert(m2);
    assert(mat_same_shape(m1, m2));

    uint32_t count = m1->rows * m1->cols;

    for (uint32_t i = 0; i < count; ++i)
        m1->values[i] += m2->values[i];
}

void mat_sub_inplace(Matrix* restrict m1, const Matrix* restrict m2) {
    mat_assert(m1);
    mat_assert(m2);
    assert(mat_same_shape(m1, m2));

    uint32_t count = m1->rows * m1->cols;

    for (uint32_t i = 0; i < count; ++i)
        m1->values[i] -= m2->values[i];
}

void mat_mulf_inplace(Matrix* restrict m, float f) {
    mat_assert(m);

    uint32_t count = m->rows * m->cols;

    for (uint32_t i = 0; i < count; ++i)
        m->values[i] *= f;
}

void mat_transpose_into(Matrix* restrict dest, const Matrix* restrict src) {
    mat_assert(dest);
    mat_assert(src);

    mat_resize(dest, src->cols, src->rows);

    for (uint32_t c = 0; c < src->cols; ++c)
        for (uint32_t r = 0; r < src->rows; ++r)
            *mat_at(dest, c, r) = mat_at_const(src, r, c);
}

void mat_mul_into(Matrix* restrict dest, const Matrix* restrict m1, const Matrix* restrict m2) {
    mat_assert(dest);
    mat_assert(m1);
    mat_assert(m2);

    assert(m1->cols == m2->rows);

    mat_resize(dest, m1->rows, m2->cols);

    uint32_t dest_index = 0;

    for (uint32_t k = 0; k < m2->cols; ++k) {
        for (uint32_t i = 0; i < m1->rows; ++i) {
            dest->values[dest_index] = 0.0f;

            for (uint32_t j = 0; j < m1->cols; ++j)
                dest->values[dest_index] += mat_at_const(m1, i, j) * mat_at_const(m2, j, k);

            dest_index++;
        }
    }
}

void mat_mul_accumulate_into(Matrix* restrict dest, const Matrix* restrict m1, const Matrix* restrict m2) {
    mat_assert(dest);
    mat_assert(m1);
    mat_assert(m2);

    assert(m1->cols == m2->rows);

    mat_resize(dest, m1->rows, m2->cols);

    uint32_t dest_index = 0;

    for (uint32_t k = 0; k < m2->cols; ++k) {
        for (uint32_t i = 0; i < m1->rows; ++i) {
            for (uint32_t j = 0; j < m1->cols; ++j)
                dest->values[dest_index] += mat_at_const(m1, i, j) * mat_at_const(m2, j, k);

            dest_index++;
        }
    }
}

void mat_mul_transposed_into(Matrix* restrict dest, const Matrix* restrict m1_T, const Matrix* restrict m2) {
    mat_assert(dest);
    mat_assert(m1_T);
    mat_assert(m2);

    assert(m1_T->rows == m2->rows);

    mat_resize(dest, m1_T->cols, m2->cols);

    uint32_t dest_index = 0;

    for (uint32_t k = 0; k < m2->cols; ++k) {
        for (uint32_t i = 0; i < m1_T->cols; ++i) {
            dest->values[dest_index] = 0.0f;

            for (uint32_t j = 0; j < m1_T->rows; ++j)
                dest->values[dest_index] += mat_at_const(m1_T, j, i) * mat_at_const(m2, j, k);

            dest_index++;
        }
    }
}