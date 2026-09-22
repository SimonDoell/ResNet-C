#include "matrix.h"

Matrix mat_new_tensor(uint32_t rows, uint32_t cols, uint32_t batches) {
    assert(rows    > 0);
    assert(cols    > 0);
    assert(batches > 0);
    
    return (Matrix){
        .rows    = rows,
        .cols    = cols,
        .batches = batches,
        .values  = calloc(rows * cols * batches, sizeof(float)),
    };
}

Matrix mat_new_matrix(uint32_t rows, uint32_t cols) {
    return mat_new_tensor(rows, cols, 1);
}

Matrix mat_new_vector(uint32_t rows) {
    return mat_new_tensor(rows, 1, 1);
}

void mat_assert(const Matrix* restrict m) {
    assert(m);
    assert(m->values);
    assert(m->rows    > 0);
    assert(m->cols    > 0);
    assert(m->batches > 0);
}

void mat_free(Matrix* restrict m) {
    mat_assert(m);
    free(m->values);

    m->values  = NULL;
    m->rows    = 0;
    m->cols    = 0;
    m->batches = 0;
}

uint32_t mat_index(uint32_t r, uint32_t c, uint32_t b, uint32_t rows, uint32_t cols) {
    return (r) + (c * rows) + (b * rows * cols); // column-major!
}

float mat_at_const(const Matrix* restrict m, uint32_t r, uint32_t c, uint32_t b) {
    mat_assert(m);
    assert(r < m->rows);
    assert(c < m->cols);
    assert(b < m->batches);

    return m->values[mat_index(r, c, b, m->rows, m->cols)]; // column-major!
}

float* mat_at(Matrix* restrict m, uint32_t r, uint32_t c, uint32_t b) {
    mat_assert(m);
    assert(r < m->rows);
    assert(c < m->cols);
    assert(b < m->batches);

    return &m->values[mat_index(r, c, b, m->rows, m->cols)]; // column-major!
}

Matrix mat_copy(const Matrix* restrict m) {
    mat_assert(m);
    Matrix result = mat_new_tensor(m->rows, m->cols, m->batches);
    memcpy(result.values, m->values, sizeof(float) * m->rows * m->cols * m->batches);
    return result;
}

void mat_move_into(Matrix* restrict dest, Matrix* restrict src) {
    assert(dest != src);
    mat_assert(dest);
    mat_assert(src);

    mat_free(dest);

    dest->rows    = src->rows;
    dest->cols    = src->cols;
    dest->batches = src->batches;
    dest->values  = src->values;

    src->rows     = 0;
    src->cols     = 0;
    src->batches  = 0;
    src->values   = NULL;
}

void mat_resize(Matrix* restrict m, uint32_t rows, uint32_t cols, uint32_t batches) {
    mat_assert(m);
    assert(rows    > 0);
    assert(cols    > 0);
    assert(batches > 0);

    if (m->rows == rows && m->cols == cols && m->batches == batches) return;
    
    uint32_t old_count = m->rows * m->cols * m->batches;
    uint32_t new_count =    rows *    cols *    batches;

    m->values = realloc(m->values, new_count * sizeof(float));
    assert(m->values);

    m->rows    = rows;
    m->cols    = cols;
    m->batches = batches;

    for (uint32_t i = old_count; i < new_count; ++i)
        m->values[i] = 0.0f;
}

bool mat_same_shape(const Matrix* restrict m1, const Matrix* restrict m2) {
    assert(m1 != m2);
    mat_assert(m1);
    mat_assert(m2);
    return (m1->rows == m2->rows) && (m1->cols == m2->cols) && (m1->batches == m2->batches);
}

void mat_add_inplace(Matrix* restrict m1, const Matrix* restrict m2) {
    assert(m1 != m2);
    assert(mat_same_shape(m1, m2));
    mat_assert(m1);
    mat_assert(m2);

    uint32_t n = m1->rows * m1->cols * m1->batches;

    for (uint32_t i = 0; i < n; ++i)
        m1->values[i] += m2->values[i];
}

void mat_sub_inplace(Matrix* restrict m1, const Matrix* restrict m2) {
    assert(m1 != m2);
    assert(mat_same_shape(m1, m2));
    mat_assert(m1);
    mat_assert(m2);

    uint32_t n = m1->rows * m1->cols * m1->batches;

    for (uint32_t i = 0; i < n; ++i)
        m1->values[i] -= m2->values[i];
}

Matrix mat_transposed(Matrix* restrict m) {
    mat_assert(m);

    Matrix result = mat_new_tensor(m->cols, m->rows, m->batches);

    for (uint32_t b = 0; b < m->batches; ++b) {
        for (uint32_t c = 0; c < m->cols; ++c) {
            for (uint32_t r = 0; r < m->rows; ++r) {
                *mat_at(&result, c, r, b) = mat_at_const(m, r, c, b);
            }
        }
    }

    return result;
}

Matrix mat_mul(const Matrix* restrict m1, const Matrix* restrict m2) {
    mat_assert(m1);
    mat_assert(m2);
    assert(m1 != m2);
    assert(m1->cols    == m2->rows);
    assert(m1->batches == m2->batches);

    uint32_t batches = m1->batches;

    Matrix result = mat_new_tensor(m1->rows, m2->cols, batches);

    for (uint32_t b = 0; b < batches; ++b) {    
        for (uint32_t i = 0; i < m1->rows; ++i) {
            for (uint32_t k = 0; k < m2->cols; ++k) {
                float sum = 0.0f;

                for (uint32_t j = 0; j < m1->cols; ++j)
                    sum += mat_at_const(m1, i, j, b) * mat_at_const(m2, j, k, b);

                *mat_at(&result, i, k, b) = sum;
            }
        }
    }

    return result;
}

void mat_add_copy_to_batch(Matrix* restrict m1, const Matrix* restrict m2) {
    mat_assert(m1);
    mat_assert(m2);
    assert(m1 != m2);
    assert(m1->rows == m2->rows);
    assert(m1->cols == m2->cols);

    mat_resize(m1, m1->rows, m1->cols, m1->batches + m2->batches);

    uint32_t base_index = mat_index(0, 0, m1->batches - m2->batches, m1->rows, m1->cols);
    uint32_t count      = (m1->rows * m1->cols) * m2->batches;

    for (int i = 0; i < count; ++i)
        m1->values[base_index + i] = m2->values[i];
}