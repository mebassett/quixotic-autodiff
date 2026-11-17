// daft_inputs.h
#ifndef DAFT_INPUTS_H
#define DAFT_INPUTS_H

#include "daft_autodiff.h"

namespace DA {

  void Input_computeGrad(Function* f, float* seed, Operation& op);
  void Weights_computeGrad(Function* f, float* seed, Operation& op);

}

#endif /* DAFT_INPUTS_H */
