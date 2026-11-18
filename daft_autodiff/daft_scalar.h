// daft_scalar.h
#ifndef DAFT_SCALAR_H
#define DAFT_SCALAR_H

#include "daft_autodiff.h"

namespace DA {

  void Scalar_computeGrad(Function* f, float* seed, Operation& op);
  void Scalar_compute(Function* f, Operation& op);

}

#endif /* DAFT_SCALAR_H */
