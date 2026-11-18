// daft_convolution.h
#ifndef DAFT_CONVOLUTION_H
#define DAFT_CONVOLUTION_H

#include "daft_autodiff.h"

namespace DA {

  void Convolution_computeGrad(Function* f, float* seed, Operation& op);
  void Convolution_compute(Function* f, Operation& op);

}

#endif /* DAFT_CONVOLUTION_H */
