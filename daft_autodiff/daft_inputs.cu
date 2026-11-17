#include "daft_autodiff.h"
#include "daft_error.h"
#include "daft_inputs.h"

using namespace DA;

namespace DA {

  Operation Operation::column(string name, uint rows) {
      return { .opType=OperationType::InputColumn
             , .workingSize = 0
             , .resultSize = rows
             , .gradSize = rows
             , .rows = rows
             , .cols = 1
             , .name{name} };
  }

  Operation Operation::inputMatrix(string name, uint rows, uint cols) {
      return { .opType=OperationType::InputMatrix
             , .workingSize = 0
             , .resultSize = rows*cols
             , .gradSize = rows*cols
             , .rows = rows
             , .cols = cols
             , .name{name} };
            
  }

  Operation Operation::weightsMatrix(string name, uint rows, uint cols) {
      return { .opType=OperationType::WeightsMatrix
             , .workingSize = 0
             , .resultSize = rows*cols
             , .gradSize = rows*cols
             , .rows = rows
             , .cols = cols
             , .name{name} };
            
  }

  void Input_computeGrad(Function* f, float* seed, Operation& op){
    float alpha = 1;
    float *grad = f->getGrad(op.name);
    cublasErrCk( cublasSaxpy(*f->cublasH, op.gradSize*f->batchSize, &alpha, seed, 1, grad, 1) );
  }

  void Weights_computeGrad(Function* f, float* seed, Operation& op){
    float alpha = 1;
    float *grad = f->getGrad(op.name);
    // we don't batch weights, because that's silly.  so we are summing all these up
    for(int i = 0; i < f->batchSize; i++)
      cublasErrCk( cublasSaxpy(*f->cublasH, op.gradSize, &alpha, seed+i*op.gradSize, 1, grad, 1) );
  }

}
