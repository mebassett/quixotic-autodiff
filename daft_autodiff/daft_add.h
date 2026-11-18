// daft_add.h
#ifndef DAFT_ADD_H
#define DAFT_ADD_H

#include "daft_autodiff.h"

namespace DA {

  void Add_computeGrad(Function* f, float* seed, Operation& op);
  void Add_compute(Function* f, Operation& op);

}

#endif /* DAFT_ADD_H */
