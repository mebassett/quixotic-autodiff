#include "daft_autodiff.h"
#include "daft_error.h"
#include "daft_innerproduct.h"

using namespace DA;

namespace DA {

Operation Operation::innerProduct(string name, string target1, string target2,
                                  uint rows) {
  return {.opType = OperationType::InnerProduct,
          .workingSize = 0,
          .resultSize = 1,
          .gradSize = 2 * rows,
          .rows = 1,
          .cols = 1,
          .name{name},
          .config{(BinaryOpConfig){.target1{target1},
                                   .target2{target2},
                                   .targetRows = rows,
                                   .targetCols = 1}}};
}

void InnerProduct_computeGrad(Function* f, float* seed, Operation& op) {
  cublasHandle_t* cublasH = f->cublasH;
  int batchSize = f->batchSize;

  BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
  float* col1 = f->getResult(opConfig.target1);
  float* col2 = f->getResult(opConfig.target2);

  float* vec1 = f->getGrad(op.name);
  float* vec2 = vec1 + opConfig.targetRows * batchSize;

  cublasErrCk(cublasSetPointerMode(*cublasH, CUBLAS_POINTER_MODE_DEVICE));

  cublasErrCk(
      cublasScopy(*cublasH, opConfig.targetRows * batchSize, col1, 1, vec1, 1));
  cublasErrCk(
      cublasSscal(*cublasH, opConfig.targetRows * batchSize, seed, vec1, 1));

  cublasErrCk(
      cublasScopy(*cublasH, opConfig.targetRows * batchSize, col2, 1, vec2, 1));
  cublasErrCk(
      cublasSscal(*cublasH, opConfig.targetRows * batchSize, seed, vec2, 1));
  cublasErrCk(cublasSetPointerMode(*cublasH, CUBLAS_POINTER_MODE_HOST));
  f->computeGrad(opConfig.target2, vec1);
  f->computeGrad(opConfig.target1, vec2);
}

void InnerProduct_compute(Function* f, Operation& op) {
  BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
  float* d_v1 = f->getResult(opConfig.target1);
  float* d_v2 = f->getResult(opConfig.target2);
  float* d_result = f->getResult(op.name);

  float alpha = 1;
  float beta = 0;

  cublasErrCk(cublasSgemvStridedBatched(
      *(f->cublasH), CUBLAS_OP_N, 1, opConfig.targetRows, &alpha, d_v1, 1,
      opConfig.targetRows, d_v2, 1, opConfig.targetRows, &beta, d_result, 1, 1,
      f->batchSize));
}

} // namespace DA
