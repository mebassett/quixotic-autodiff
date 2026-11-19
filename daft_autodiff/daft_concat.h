// daft_concat.h
#ifndef DAFT_CONCAT_H
#define DAFT_CONCAT_H

#include "daft_autodiff.h"

namespace DA {

void Concat_computeGrad(Function* f, float* seed, Operation& op);
void Concat_compute(Function* f, Operation& op);

} // namespace DA

#endif /* DAFT_CONCAT_H */
