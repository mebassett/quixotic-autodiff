// daft_leaky_relu.h
#ifndef DAFT_LEAKY_RELU_H
#define DAFT_LEAKY_RELU_H

#include "daft_autodiff.h"

namespace DA {

void LeakyReLU_computeGrad(Function* f, float* seed, Operation& op);
void LeakyReLU_compute(Function* f, Operation& op);

} // namespace DA

#endif /* DAFT_LEAKY_RELU_H */
