#include "daft_autodiff.h"
#include "daft_error.h"
#include "daft_maxpool.h"
#include "daft_kernels.cuh"

using namespace DA;

namespace DA {

  Operation Operation::maxPool(string name, string target, uint width, uint height, 
          uint rowSkip, uint colSkip, uint targetRows, uint targetCols) {
      uint rows = (targetRows - height) / rowSkip + 1;
      uint cols = (targetCols - width) / colSkip + 1;
      
      return { .opType=OperationType::MaxPool
             , .workingSize = 0
             , .resultSize = rows * cols
             , .gradSize = targetRows * targetCols
             , .rows = rows
             , .cols = cols
             , .name{name}
             , .config{ (MaxPoolConfig) {
                 .target{target}
               , .width = width
               , .height = height
               , .rowSkip = rowSkip
               , .colSkip = colSkip
               , .targetRows = targetRows
               , .targetCols = targetCols
               }}
             };
  }

  void MaxPool_computeGrad(Function* f, float* seed, Operation& op) {
    MaxPoolConfig opConfig = get<MaxPoolConfig>(op.config);
    float* targetValue = f->getResult(opConfig.target);
    float* result = f->getResult(op.name);
    float* grad = f->getGrad(op.name);

    dim3 bd(16, 16, 4);
    dim3 gd(ceil(op.cols / 16.0), ceil(op.rows / 16.0), ceil(f->batchSize / 4.0));
    doMaxPoolGrad<<<gd, bd>>>(op.rows, op.cols, opConfig.targetRows, opConfig.targetCols, 
        opConfig.rowSkip, opConfig.height, opConfig.colSkip, opConfig.width, 
        targetValue, result, seed, grad, f->batchSize);
    cudaErrCk( cudaPeekAtLastError() );

    f->computeGrad(opConfig.target, grad);
  }

  void MaxPool_compute(Function* f, Operation& op) {
    MaxPoolConfig opConfig = get<MaxPoolConfig>(op.config);
    float* d_target = f->getResult(opConfig.target);
    float* d_result = f->getResult(op.name);

    dim3 bd(16, 16, 4);
    dim3 gd(ceil(op.cols / 16.0), ceil(op.rows / 16.0), ceil(f->batchSize / 4.0));

    doMaxPool<<<gd, bd>>>(op.rows, op.cols, opConfig.targetRows, opConfig.targetCols,
        opConfig.rowSkip, opConfig.height, opConfig.colSkip, opConfig.width,
        d_target, d_result, f->batchSize);
    cudaErrCk( cudaPeekAtLastError() );

  }

}
