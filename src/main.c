#include <stdio.h>
#include "matrix.h"
#include "countof.h"

int main() {
    Matrix base = mat_new_tensor(2, 2, 7);
    Matrix new  = mat_new_matrix(2, 2);

    mat_add_copy_to_batch(&base, &new);

    printf("Final structure: %i %i %i\n\n", base.rows, base.cols, base.batches);
    
    return 0;
}