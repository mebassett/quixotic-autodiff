#include "daft_add.h"
#include "daft_autodiff.h"
#include "daft_error.h"

using namespace DA;

namespace DA {

Operation Operation::add(string name, string target1, string target2, uint rows,
                         uint cols) {
  return {.opType = OperationType::Add,
          .workingSize = 0,
          .resultSize = rows * cols,
          .gradSize = rows * cols,
          .rows = rows,
          .cols = cols,
          .name{name},
          .config{(BinaryOpConfig){.target1{target1},
                                   .target2{target2},
                                   .targetRows = rows,
                                   .targetCols = 1}}};
}

void Add_computeGrad(Function* f, float* seed, Operation& op) {
  BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
  float* copySeed = f->getGrad(op.name);

  cudaMemcpy(copySeed, seed, f->batchSize * op.rows * op.cols * sizeof(float),
             cudaMemcpyDeviceToDevice);
  f->computeGrad(opConfig.target1, seed);
  f->computeGrad(opConfig.target2, copySeed);
}

void Add_compute(Function* f, Operation& op) {
  BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
  float* d_v1 = f->getResult(opConfig.target1);
  float* d_v2 = f->getResult(opConfig.target2);
  float* d_result = f->getResult(op.name);
  float alpha = 1.0;

  cublasErrCk(cublasScopy(*f->cublasH, op.rows * op.cols * f->batchSize, d_v1,
                          1, d_result, 1));
  cublasErrCk(cublasSaxpy(*f->cublasH, op.rows * op.cols * f->batchSize, &alpha,
                          d_v2, 1, d_result, 1));
}

} // namespace DA
