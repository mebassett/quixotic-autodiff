#include "daft_autodiff.h"
#include "daft_error.h"
#include "daft_convolution.h"
#include "daft_kernels.cuh"

using namespace DA;

void convolutionPadInput( uint inputRows, uint inputCols
                        , uint rowPadding, uint colPadding
                        , float* input, float* output, int batchSize) {
    uint outputRows = inputRows + rowPadding*2;
    uint outputCols = inputCols + colPadding*2;
    dim3 gd(ceil(outputCols / 16.0), ceil(outputRows / 16.0),  ceil(batchSize / 4.0));
    dim3 bd(16, 16, 4);

    doPadInput<<<gd, bd>>>(input, output, inputRows
                , inputCols, rowPadding, colPadding, batchSize);
    cudaErrCk( cudaPeekAtLastError() );
    cudaDeviceSynchronize(); //do you actually need this? 
}

void convolutionUnrollKernel( uint unrKrnlRows, uint unrKrnlCols
                            , uint kernelRows, uint kernelCols
                            , uint paddedInputCols
                            , uint outputCols
                            , uint rowSkip
                            , uint colSkip
                            , float* kernel
                            , float* output ) {
    dim3 gd(ceil(unrKrnlCols / 32.0), ceil(unrKrnlRows / 32.0), 1);
    dim3 bd(32, 32, 1);
    doUnroll<<<gd, bd>>>( kernel
                        , output
                        , kernelRows
                        , kernelCols
                        , unrKrnlRows
                        , unrKrnlCols
                        , paddedInputCols
                        , outputCols
                        , rowSkip
                        , colSkip );
    cudaErrCk( cudaPeekAtLastError() );
    cudaDeviceSynchronize(); //do you actually need this? 
}

namespace DA {
  Operation Operation::convolution(string name, string multiplicand, string kernel, uint rowPadding,
          uint rowSkip, uint colPadding, uint colSkip, 
          uint multiplicandRows, uint multiplicandCols,
          uint kernelRows, uint kernelCols) {
      int rows = (multiplicandRows + 2 * rowPadding - kernelRows) / rowSkip + 1;
      int cols = (multiplicandCols + 2 * colPadding - kernelCols) / colSkip + 1;

      int unrKrnlCols = (multiplicandRows + rowPadding*2) * (multiplicandCols + colPadding*2);
      int unrKrnlRows = rows * cols; 

      // working size ihe padding input plus the unrolled kernel.
      int paddedInputSize = 
        (multiplicandRows + 2 * rowPadding) * (multiplicandCols + 2 * colPadding);

      // grad size will be 
      //    size of the original kernel matrix
      //  + size of the original input matrix (multiplicand)
      //  BUT we also need some working memory for the unrolling and unpadding
      //  so we also have
      //  + size of unrolled kernel matrix
      //  + size of a column of the unrolled kernel matrix
      // of the multiplicand plus
      int gradSize =    (kernelRows * kernelCols) 
                      + (multiplicandRows * multiplicandCols)
                      + (unrKrnlRows * unrKrnlCols)
                      + unrKrnlCols;
      
      return { .opType=OperationType::Convolution
             , .workingSize = paddedInputSize + (unrKrnlRows * unrKrnlCols)
             , .resultSize = rows * cols
             , .gradSize = gradSize
             , .rows = rows
             , .cols = cols
             , .name{name}
             , .config{ (ConvolutionConfig) {
                 .multiplicand{multiplicand}
               , .kernel{kernel}
               , .rowPadding = rowPadding
               , .rowSkip = rowSkip
               , .colPadding = colPadding
               , .colSkip = colSkip
               , .multiplicandRows = multiplicandRows
               , .multiplicandCols = multiplicandCols
               , .kernelRows = kernelRows
               , .kernelCols = kernelCols
               , .unrKrnlRows = unrKrnlRows
               , .unrKrnlCols = unrKrnlCols
               , .paddedInputSize = paddedInputSize
               }}
             };
  }

  void Convolution_computeGrad(Function* f, float* seed, Operation& op) {
    ConvolutionConfig opCnfg = get<ConvolutionConfig>(op.config);

    float alpha = 1;
    float beta = 0;

    int startInput = opCnfg.kernelRows * opCnfg.kernelCols * f->batchSize;
    int startMatrix = startInput 
        + (opCnfg.multiplicandRows * opCnfg.multiplicandCols * f->batchSize );
    int startCol = startMatrix
        + (opCnfg.unrKrnlRows * opCnfg.unrKrnlCols )*f->batchSize;

    float* rolledKernelMatrixGrad = f->getGrad(op.name);
    float* inputGrad = f->getGrad(op.name) + startInput;
    float* matrixGrad = f->getGrad(op.name) + startMatrix; 
    float* colGrad = f->getGrad(op.name) + startCol; 

    float* paddedInput = f->getWorkingMemory(op.name);
    float* unrolledKernel = f->getWorkingMemory(op.name)
                        + opCnfg.paddedInputSize * f->batchSize;

    int matrixGradSize = opCnfg.unrKrnlCols * opCnfg.unrKrnlRows;
    int paddedInputSize = opCnfg.unrKrnlCols;
    int seedSize = opCnfg.unrKrnlRows;


    cublasErrCk( cublasSgemmStridedBatched( *f->cublasH
                , CUBLAS_OP_T
                , CUBLAS_OP_N
                , opCnfg.unrKrnlCols
                , opCnfg.unrKrnlRows
                , 1
                , &alpha
                , paddedInput
                , 1
                , paddedInputSize
                , seed
                , 1
                , seedSize
                , &beta
                , matrixGrad
                , opCnfg.unrKrnlCols
                , matrixGradSize //stride matrixGrad
                , f->batchSize));
 
    dim3 bd(32, 1, 32);
    dim3 gd(ceil(opCnfg.unrKrnlCols / 32.0), 1, ceil(f->batchSize / 32.0));

        
    doKernelRoll<<<gd, bd>>>( matrixGrad
        , rolledKernelMatrixGrad
        , opCnfg.kernelRows
        , opCnfg.kernelCols
        , opCnfg.unrKrnlCols
        , opCnfg.unrKrnlRows
        , opCnfg.colSkip
        , opCnfg.rowSkip
        , opCnfg.multiplicandCols + 2 * opCnfg.colPadding
        , op.cols
        , f->batchSize);
    cudaErrCk( cudaPeekAtLastError() );



    f->computeGrad(opCnfg.kernel, rolledKernelMatrixGrad);


    cublasErrCk(
      cublasSgemmStridedBatched( *f->cublasH
          , CUBLAS_OP_N
          , CUBLAS_OP_T
          , 1
          , opCnfg.unrKrnlCols
          , opCnfg.unrKrnlRows
          , &alpha
          , seed
          , 1
          , seedSize
          , unrolledKernel
          , opCnfg.unrKrnlCols
          , 0 //. stride unrolledKernel
          , &beta
          , colGrad
          , 1
          , opCnfg.unrKrnlCols // stride colGrad
          , f->batchSize)
    );


    dim3 gd2( ceil(opCnfg.multiplicandCols / 16.0)
            , ceil(opCnfg.multiplicandRows / 16.0)
            , ceil(f->batchSize / 4.0));
    dim3 bd2(16, 16, 4);
    doCopyWithoutPadding<<<gd2, bd2>>>( colGrad
            , inputGrad
            , opCnfg.multiplicandRows
            , opCnfg.multiplicandCols
            , opCnfg.rowPadding
            , opCnfg.colPadding
            , f->batchSize);
    cudaErrCk( cudaPeekAtLastError() );
    f->computeGrad(opCnfg.multiplicand, inputGrad);
  }

  void Convolution_compute(Function* f, Operation& op) {
    // pad the input, which means a copy to wocrking, which is slow
    ConvolutionConfig opCnfg = get<ConvolutionConfig>(op.config);
    float* input = f->getResult(opCnfg.multiplicand);
    float* paddedInput = f->getWorkingMemory(op.name);
    convolutionPadInput( opCnfg.multiplicandRows
                       , opCnfg.multiplicandCols
                       , opCnfg.rowPadding
                       , opCnfg.colPadding
                       , input
                       , paddedInput
                       , f->batchSize);

    // unroll the kernel..another slow copy to working.
    float* kernel = f->getResult(opCnfg.kernel);
    float* unrolledKernel = f->getWorkingMemory(op.name)
                            + opCnfg.paddedInputSize * f->batchSize;
    convolutionUnrollKernel( opCnfg.unrKrnlRows
            , opCnfg.unrKrnlCols
            , opCnfg.kernelRows
            , opCnfg.kernelCols
            , opCnfg.multiplicandCols + 2 * opCnfg.colPadding
            , op.cols
            , opCnfg.rowSkip
            , opCnfg.colSkip
            , kernel
            , unrolledKernel);

    // do a matrix multiplication, storing it in the result
    float alpha = 1;
    float beta = 0;
    float* output = f->getResult(op.name);

    cublasErrCk( cublasSgemvStridedBatched( *f->cublasH
                , CUBLAS_OP_T
                , opCnfg.unrKrnlCols
                , opCnfg.unrKrnlRows
                , &alpha
                , unrolledKernel // As
                , opCnfg.unrKrnlCols
                , 0 // stride A
                , paddedInput // x
                , 1
                , opCnfg.paddedInputSize // stride x
                , &beta
                , output
                , 1
                , op.resultSize
                , f->batchSize ) );

  }
}
