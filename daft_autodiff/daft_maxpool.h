// daft_maxpool.h
#ifndef DAFT_MAXPOOL_H
#define DAFT_MAXPOOL_H

#include "daft_autodiff.h"

namespace DA {

  void MaxPool_computeGrad(Function* f, float* seed, Operation& op);
  void MaxPool_compute(Function* f, Operation& op);

}

#endif /* DAFT_MAXPOOL_H */
