// daft_matrix_product.h
#ifndef DAFT_MATRIX_PRODUCT_H
#define DAFT_MATRIX_PRODUCT_H

#include "daft_autodiff.h"

namespace DA {

  void MatrixProduct_computeGrad(Function* f, float* seed, Operation& op);
  void MatrixProduct_compute(Function* f, Operation& op);

}

#endif /* DAFT_MATRIX_PRODUCT_H */
