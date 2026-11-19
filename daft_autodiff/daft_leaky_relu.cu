#include "daft_autodiff.h"
#include "daft_error.h"
#include "daft_kernels.cuh"
#include "daft_leaky_relu.h"

using namespace DA;

namespace DA {

Operation Operation::applyLeakyReLU(string name, string target, uint rows,
                                    uint cols) {
  BasicConfig config = {.target{target}};
  return {.opType = OperationType::LeakyReLU,
          .workingSize = rows * cols,
          .resultSize = rows * cols,
          .gradSize = rows * cols,
          .rows = rows,
          .cols = cols,
          .name{name},
          .config{(BasicConfig){.target{target}}}};
}

void LeakyReLU_computeGrad(Function* f, float* seed, Operation& op) {
  BasicConfig opConfig = get<BasicConfig>(op.config);
  float* newSeed = f->getWorkingMemory(op.name);
  float* grad = f->getGrad(op.name);
  dim3 bd(16, 16, 4);
  dim3 gd(ceil(op.cols / 16.0), ceil(op.rows / 16.0), ceil(f->batchSize / 4.0));

  doComponentProduct<<<gd, bd>>>(op.rows, op.cols, grad, seed, newSeed,
                                 f->batchSize);

  f->computeGrad(opConfig.target, newSeed);
}

void LeakyReLU_compute(Function* f, Operation& op) {
  BasicConfig opConfig = get<BasicConfig>(op.config);
  float* d_col = f->getResult(opConfig.target);
  float* d_grad = f->getGrad(op.name);
  float* d_result = f->getResult(op.name);

  dim3 bd(16, 16, 4);
  dim3 gd(ceil(op.cols / 16.0), ceil(op.rows / 16.0), ceil(f->batchSize / 4.0));

  doLeakyReLU<<<gd, bd>>>(op.rows, op.cols, d_grad, d_col, d_result,
                          f->batchSize);
  cudaErrCk(cudaPeekAtLastError());
}

} // namespace DA
