// daft_kernels.cuh
#ifndef DAFT_KERNELS_CUH
#define DAFT_KERNELS_CUH

namespace DA {
  __global__ void doBatchedSaxpy(int colSize, int batchSize, float* seed, float* grad) ;
  __global__ void doLeakyReLU(int Arows, int Acols, float* grad, float* A, float* result, int batchSize);
  void setConcatSizes(int* targetSizes, int size);
  __global__ void doConcatInterleave(float** targets, int numTargets, float* result, int resultSize, int ) ;
  __global__ void doConcatDeinterleave(float* inputSeed, int numTargets, float* result, int resultSize, int batchSize) ;
  __global__ void doFill(int rows, int cols, float value, float* result);
  __global__ void doComponentProduct(int rows, int cols, float* grad, float* seed, float* result, int batchSize);
  __global__ void doPadInput(float* input, float* paddedInput, int inputRows, int inputCols, int rowPadding, int colPadding, int batchSize);

  __global__ void doUnroll(float* kernel, float* matrix, int kernelRows,
      int kernelCols, int mRows, int mCols, int inCols,
      int outCols, int rowSkip, int colSkip);

  __global__ void doKernelRoll(float* matrix, float* kernel, int kernelRows,
      int kernelCols, int mCols, int mRows, int colSkip,
      int rowSkip, int inCols, int outCols, int batchSize);

  __global__ void doCopyWithoutPadding(float* paddedSource, float* dest, int rows,
      int cols, int rowPadding, int colPadding, int batchSize);

  __global__ void doMaxPool(int targetRows, int targetCols, int matrixRows,
      int matrixCols, int rowSkip, int height, int colSkip,
      int width, float* matrix, float* result, int batchSize);

  __global__ void doMaxPoolGrad(int targetRows, int targetCols, int matrixRows,
      int matrixCols, int rowSkip, int height,
      int colSkip, int width, float* matrix,
      float* value, float* seed, float* result, int batchSize);

}   

#endif /* DAFT_KERNELS_CUH */
