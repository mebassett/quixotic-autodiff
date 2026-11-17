// daft_innerproduct.h
#ifndef DAFT_INNERPRODUCT_H
#define DAFT_INNERPRODUCT_H

#include "daft_autodiff.h"

namespace DA {

  void InnerProduct_computeGrad(Function* f, float* seed, Operation& op);
  void InnerProduct_compute(Function* f, Operation& op);

}

#endif /* DAFT_INNERPRODUCT_H */
