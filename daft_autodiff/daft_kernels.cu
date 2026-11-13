#include "daft_kernels.cuh"
#include "daft_error.h"

namespace DA {

 __global__ void doBatchedSaxpy(int colSize, int batchSize, float* seed, float* grad) {
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int resultIdx = blockIdx.z * blockDim.z + threadIdx.z;


     if( row < colSize && resultIdx < batchSize) {
         int idx = resultIdx * colSize + row; 
         grad[idx] = grad[idx] + seed[idx];
     }
 }

 //TODO move to cuda file shared between both silly and daft things.
 __global__ void doLeakyReLU(int Arows, int Acols, float* grad, float* A,
     float* result, int batchSize)
 {
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     int resultIdx = blockIdx.z * blockDim.z + threadIdx.z;

     if ((row < Arows) && (col < Acols) && resultIdx < batchSize) {
         int i = row * Acols + col + resultIdx * Acols * Arows;
         if (A[i] >= 0) {
             grad[i] = 1;
             result[i] = A[i];
         } else {
             result[i] = 0.01 * A[i];
             grad[i] = 0.01;
         }
     }
 }

 __constant__ int concatSizes[32];
 void setConcatSizes(int* targetSizes, int size) {
   cudaMemcpyToSymbol(concatSizes, targetSizes, size * sizeof(int));
 }

 __global__ void doConcatInterleave(float** targets, int numTargets, float* result, int resultSize, int batchSize) 
 {
     int elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
     int targetIdx = blockIdx.y * blockDim.y + threadIdx.y;
     int batchIdx = blockIdx.z * blockDim.z + threadIdx.z;

     int sumSizesTilNow = 0;
     // very unfriendky to cuda kernels...
     for(int i =0;i<targetIdx;i++)
         sumSizesTilNow += concatSizes[i];


     if(targetIdx < numTargets 
             && batchIdx < batchSize 
             && elemIdx < concatSizes[targetIdx] // hmm, not very cuda kernel friendly 

       ) {
         int idx = batchIdx * resultSize + sumSizesTilNow + elemIdx;

         result[idx] = targets[targetIdx][elemIdx + batchIdx * concatSizes[targetIdx]];

     }
 }
 __global__ void doConcatDeinterleave(float* inputSeed, int numTargets, float* result, int resultSize, int batchSize) {
     int elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
     int targetIdx = blockIdx.y * blockDim.y + threadIdx.y;
     int batchIdx = blockIdx.z * blockDim.z + threadIdx.z;


     int sumSizesTilNow = 0;
     // very unfriendky to cuda kernels...
     for(int i =0; i<targetIdx;i++)
         sumSizesTilNow += concatSizes[i];

     int inputIdx = elemIdx + sumSizesTilNow + batchIdx * resultSize;


     int resultIdx = elemIdx + batchIdx * concatSizes[targetIdx] + batchSize * sumSizesTilNow;
     if(targetIdx < numTargets 
             && batchIdx < batchSize 
             && elemIdx < concatSizes[targetIdx] // hmm, not very cuda kernel friendly 

       ) {
       result[resultIdx] = inputSeed[inputIdx];
     }


 }

 __global__ void doFill(int rows, int cols, float value, float* result)
 {
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     int i = row * cols + col;
     if (i < rows * cols)
         result[i] = value;
 }

 __global__ void doComponentProduct(int rows, int cols, float* grad, float* seed,
     float* result, int batchSize)
 {
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     int resultIdx = blockIdx.z * blockDim.z + threadIdx.z;
     if (row < rows && col < cols && resultIdx < batchSize) {
         int i = (rows * cols * resultIdx) + (row * cols) + col; 
         result[i] = seed[i] * grad[i];
     }
 }

 __global__ void doPadInput(float* input, float* paddedInput, int inputRows,
     int inputCols, int rowPadding, int colPadding, int batchSize)
 {
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     int resultIdx = blockIdx.z * blockDim.z + threadIdx.z;

     int rows = inputRows + 2 * rowPadding;
     int cols = inputCols + 2 * colPadding;
 
     if (row < rows && col < cols && resultIdx < batchSize) {
         int outIdx = rows * cols * resultIdx + row*cols + col;
         int inIdx = inputCols * inputRows * resultIdx + (row - rowPadding) * inputCols + col - colPadding; 

         if (row - rowPadding >= 0 && row < inputRows + rowPadding && col - colPadding >= 0 && col < inputCols + colPadding)
             paddedInput[outIdx] = input[inIdx];
         else
             paddedInput[outIdx] = 0;
     }
 }

 __global__ void doUnroll(float* kernel, float* matrix, int kernelRows,
     int kernelCols, int mRows, int mCols, int inCols,
     int outCols, int rowSkip, int colSkip)
 {
     int mrow = blockIdx.y * blockDim.y + threadIdx.y;
     int mcol = blockIdx.x * blockDim.x + threadIdx.x;
     if (mrow < mRows && mcol < mCols) {
         int outRow = mrow / outCols;
         int outCol = mrow % outCols;
 
         int inRow = mcol / inCols;
         int inCol = mcol % inCols;
 
         int kRowIndex = inRow - rowSkip * outRow;
         int kColIndex = inCol - colSkip * outCol;
 
         if (kRowIndex >= 0 && kRowIndex < kernelRows && kColIndex >= 0 && kColIndex < kernelCols) {
             matrix[mrow * mCols + mcol] = kernel[kRowIndex * kernelCols + kColIndex];
         } else {
             matrix[mrow * mCols + mcol] = 0;
         }
     }
 }

 __global__ void doKernelRoll(float* matrix, float* kernel, int kernelRows,
     int kernelCols, int mCols, int mRows, int colSkip,
     int rowSkip, int inCols, int outCols, int batchSize)
 {
     int mcol = blockIdx.x * blockDim.x + threadIdx.x;
     int resultIdx = blockIdx.z * blockDim.z + threadIdx.z;
     const int matrixSize = mCols * mRows;
     if (mcol < mCols && resultIdx < batchSize) {
         int kRow = mcol / inCols;
         int kCol = mcol % inCols;
         if (kRow >= 0 && kRow < kernelRows && kCol >= 0 && kCol < kernelCols) {
             float val = 0;
             
             for (int mrow = 0; mrow < mRows; mrow++) {
                 int ocol = mrow % outCols;
                 int orow = mrow / outCols;
 
                 int offset = colSkip * ocol + rowSkip * orow * inCols;

                 int matrixIdx = mrow * mCols + mcol + offset;
 
                 val += matrix[matrixIdx + resultIdx * matrixSize];
             }
             kernel[kRow * kernelCols + kCol + resultIdx * kernelCols * kernelRows] = val;
         }
     }
 }

 __global__ void doCopyWithoutPadding(float* paddedSource, float* dest, int rows,
     int cols, int rowPadding, int colPadding, int batchSize)
 {
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     int resultIdx = blockIdx.z * blockDim.z + threadIdx.z;
     int destSize = rows * cols;
     int srcSize = (rows+2*rowPadding)*(cols+2*colPadding);
     if (row < rows && col < cols && resultIdx < batchSize) {
         dest[row * cols + col + resultIdx * destSize] = paddedSource[(row + rowPadding) * (cols + 2 * colPadding) + col + colPadding + resultIdx * srcSize];
     }
 }

 __global__ void doMaxPool(int targetRows, int targetCols, int matrixRows,
     int matrixCols, int rowSkip, int height, int colSkip,
     int width, float* matrix, float* result, int batchSize)
 {
     int trow = blockIdx.y * blockDim.y + threadIdx.y;
     int tcol = blockIdx.x * blockDim.x + threadIdx.x;
     int resultIdx = blockIdx.z * blockDim.z + threadIdx.z;

     if (trow < targetRows && tcol < targetCols && resultIdx < batchSize) {
         float val = -9999;
         int mrow = rowSkip * trow;
         int mcol = colSkip * tcol;

         for (int i = mrow; i < mrow + height; i++)
             for (int j = mcol; j < mcol + width; j++)
                 if (i >= 0 && j >= 0 && i < matrixRows && j < matrixCols) {
                     int mIndex = matrixCols * i + j + resultIdx * matrixRows * matrixCols;
                     if (matrix[mIndex] > val)
                         val = matrix[mIndex];
                 }

         result[targetCols * trow + tcol + resultIdx * targetRows * targetCols] = val;
     }
 }

 __global__ void doMaxPoolGrad(int targetRows, int targetCols, int matrixRows,
     int matrixCols, int rowSkip, int height,
     int colSkip, int width, float* matrix,
     float* value, float* seed, float* result, int batchSize)
 {
     int trow = blockIdx.y * blockDim.y + threadIdx.y;
     int tcol = blockIdx.x * blockDim.x + threadIdx.x;
     int resultIdx = blockIdx.z * blockDim.z + threadIdx.z;
     int tIndex = targetCols * trow + tcol + resultIdx * targetRows * targetCols;
     if (trow < targetRows && tcol < targetCols && resultIdx < batchSize) {
         int mrow = rowSkip * trow;
         int mcol = colSkip * tcol;

         float val = seed[tIndex];

         for (int i = mrow; i < mrow + height; i++)
             for (int j = mcol; j < mcol + width; j++)
                 if (i >= 0 && j >= 0 && i < matrixRows && j < matrixCols) {
                     int mIndex = matrixCols * i + j + (resultIdx * matrixRows * matrixCols);
                     result[mIndex] = val * (matrix[mIndex] == value[tIndex]);
                 }
     }
 }

}
