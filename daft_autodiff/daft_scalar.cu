#include "daft_autodiff.h"
#include "daft_error.h"
#include "daft_scalar.h"

using namespace DA;

namespace DA {

Operation Operation::scalarMultiply(string name, string target, uint rows,
                                    uint cols, float scale) {
  return {.opType = OperationType::Scalar,
          .workingSize = 0,
          .resultSize = rows * cols,
          .gradSize = rows * cols,
          .rows = rows,
          .cols = cols,
          .name{name},
          .config{(ScalarConfig){.target{target}, .scale = scale}}};
}

void Scalar_computeGrad(Function* f, float* seed, Operation& op) {
  ScalarConfig opConfig = get<ScalarConfig>(op.config);
  float* result = f->getResult(op.name);
  float* grad = f->getGrad(op.name);

  cublasErrCk(cublasScopy(*f->cublasH, op.cols * op.rows * f->batchSize, seed,
                          1, grad, 1));
  cublasErrCk(cublasSscal(*f->cublasH, op.cols * op.rows * f->batchSize,
                          &(opConfig.scale), grad, 1));
  f->computeGrad(opConfig.target, grad);
}

void Scalar_compute(Function* f, Operation& op) {
  ScalarConfig opConfig = get<ScalarConfig>(op.config);
  float* d_target = f->getResult(opConfig.target);
  float* d_result = f->getResult(op.name);
  cublasErrCk(cublasScopy(*f->cublasH, op.rows * op.cols * f->batchSize,
                          d_target, 1, d_result, 1));
  cublasErrCk(cublasSscal(*f->cublasH, op.rows * op.cols * f->batchSize,
                          &(opConfig.scale), d_result, 1));
}

} // namespace DA
