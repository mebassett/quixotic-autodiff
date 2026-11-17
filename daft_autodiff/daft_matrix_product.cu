#include "daft_autodiff.h"
#include "daft_error.h"
#include "daft_innerproduct.h"

using namespace DA;

namespace DA {

  Operation Operation::matrixProduct(string name, string target1, string target2, uint target1Rows, uint target1Cols, uint target2Cols) {
      return { .opType=OperationType::MatrixProduct
             , .workingSize = 0 
             , .resultSize = target1Rows*target2Cols
             , .gradSize = target1Rows*target1Cols + target1Cols*target2Cols
             , .rows = target1Rows
             , .cols = target2Cols
             , .name{name} 
             , .config{ (BinaryMatrixConfig) { .target1{target1}
                                             , .target2{target2}
                                             , .target1Rows = target1Rows
                                             , .target1Cols = target1Cols
                                             , .target2Cols = target2Cols
                                             }}
             };
  } 

  void MatrixProduct_computeGrad(Function* f, float* seed, Operation& op) {
    BinaryMatrixConfig opConfig = get<BinaryMatrixConfig>(op.config);

    Operation* targetOp1;
    Operation* targetOp2;
    try {
      targetOp1 = f->opsMap.at(opConfig.target1);
      targetOp2 = f->opsMap.at(opConfig.target2);
    } catch (const out_of_range&) {
      cout << "compute(MatrixProduct) cannot find op " <<
              opConfig.target1 << " or " << opConfig.target2 << endl;

      exit(1);
      return;
    }

    float *colValue = f->getResult(opConfig.target2);//memLocs[opConfig.target2+"_result"];
    float *matrixGrad = f->getGrad(op.name);//memLocs[op.name+"_grad"];
    float *matrixValue = f->getResult(opConfig.target1);//memLocs[opConfig.target1+"_result"];
    float *colGrad = matrixGrad + f->batchSize * opConfig.target1Rows * opConfig.target1Cols;

    int matrixValueStride = targetOp1->rows * targetOp1->cols;
    if(targetOp1->opType == OperationType::WeightsMatrix)
      matrixValueStride = 0;


    float alpha = 1;
    float beta = 0;

    cublasErrCk( cublasSgemmStridedBatched( *f->cublasH
               , CUBLAS_OP_T
               , CUBLAS_OP_N
               , opConfig.target1Cols
               , opConfig.target1Rows
               , 1
               , &alpha
               , colValue
               , 1
               , targetOp2->resultSize // stride_colValues
               , seed
               , 1
               , opConfig.target1Rows // stride_seeds
               , &beta
               , matrixGrad
               , opConfig.target1Cols
               , opConfig.target1Rows * opConfig.target1Cols // stide_matrixGrads
               , f->batchSize) ); 
    
    cublasErrCk( cublasSgemmStridedBatched( *f->cublasH
               , CUBLAS_OP_N
               , CUBLAS_OP_T
               , 1
               , opConfig.target1Cols
               , opConfig.target1Rows
               , &alpha
               , seed
               , 1
               , opConfig.target1Rows // stride_seeds
               , matrixValue
               , opConfig.target1Cols
               , matrixValueStride
               , &beta
               , colGrad
               , 1
               , opConfig.target1Cols * opConfig.target2Cols // stide_colGrad
               , f->batchSize) );

    f->computeGrad(opConfig.target1, matrixGrad);
    f->computeGrad(opConfig.target2, colGrad);
  }
  
  void MatrixProduct_compute(Function* f, Operation& op) {
    BinaryMatrixConfig opConfig = get<BinaryMatrixConfig>(op.config);

    float alpha = 1;
    float beta = 0;

    // if target1 is an Input it must not batch, all the pointers must point to the same matrixValue

    // the colValue however should be batched, meaning we can pass it in directly all the time.
    Operation* targetOp1;
    Operation* targetOp2;
    try {
      targetOp1 = f->opsMap.at(opConfig.target1);
      targetOp2 = f->opsMap.at(opConfig.target2);
    } catch (const out_of_range&) {
      cout << "compute(MatrixProduct) cannot find op " <<
              opConfig.target1 << " or " << opConfig.target2 << endl;

      exit(1);
      return;
    }

    int strideA = targetOp1->resultSize;

    if(targetOp1->opType == OperationType::InputMatrix || targetOp1->opType == OperationType::WeightsMatrix){
      strideA = 0;
    }

    float* d_A = f->getResult(opConfig.target1);//memLocs[opConfig.target1+"_result"];
    float* d_B = f->getResult(opConfig.target2);//memLocs[opConfig.target2+"_result"];
    float* d_C = f->getResult(op.name);//memLocs[op.name+"_result"];


    cublasErrCk( cublasSgemmStridedBatched( *f->cublasH
               , CUBLAS_OP_N
               , CUBLAS_OP_N
               , opConfig.target2Cols
               , opConfig.target1Rows
               , opConfig.target1Cols
               , &alpha
               , d_B
               , opConfig.target2Cols
               , targetOp2->resultSize // strideB
               , d_A
               , opConfig.target1Cols
               , strideA
               , &beta
               , d_C
               , opConfig.target2Cols
               , op.resultSize // striceC
               , f->batchSize) ) 


  }

}
