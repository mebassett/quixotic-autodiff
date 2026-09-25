#include "daft_autodiff.h"
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

namespace DA {

void noOpBatchCompute(Function* f, int idx, int length) {}

#define cudaErrCk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line) {
    if(err != cudaSuccess) {
        fprintf(stderr, "CudaAssert: %s %s %d\n", cudaGetErrorString(err), file, line);
        exit(err);
    }
}

#define cublasErrCk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t err, const char *file, int line) {
    if(err != CUBLAS_STATUS_SUCCESS) {
      const char* errMsg;
      switch(err) {
         case CUBLAS_STATUS_NOT_INITIALIZED: errMsg = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
         case CUBLAS_STATUS_ALLOC_FAILED: errMsg = "CUBLAS_STATUS_ALLOC_FAILED"; break;
         case CUBLAS_STATUS_INVALID_VALUE: errMsg = "CUBLAS_STATUS_INVALID_VALUE"; break;
         case CUBLAS_STATUS_ARCH_MISMATCH: errMsg = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
         case CUBLAS_STATUS_MAPPING_ERROR: errMsg = "CUBLAS_STATUS_MAPPING_ERROR"; break;
         case CUBLAS_STATUS_EXECUTION_FAILED: errMsg = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
         case CUBLAS_STATUS_INTERNAL_ERROR: errMsg = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
         case CUBLAS_STATUS_NOT_SUPPORTED: errMsg = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
         case CUBLAS_STATUS_LICENSE_ERROR: errMsg = "CUBLAS_STATUS_LICENSE_ERROR"; break;
         default: errMsg = "Unknown CUBLAS error"; break;
      }
      fprintf(stderr,"CUBLAS assert: %s %s %d\n", errMsg, file, line); 
      exit(err);
    }
}

    ostream& operator<<(ostream &o, const OperationType t) {
        switch(t) {
            case OperationType::InputColumn:
                o << "InputColumn";
                break;
            case OperationType::InputMatrix:
                o << "InputMatrix";
                break;
            case OperationType::MatrixProduct:
                o << "MatrixProduct";
                break;
            case OperationType::LeakyReLU:
                o << "LeakyReLU";
                break;
            case OperationType::Add:
                o << "Add";
                break;
            case OperationType::Scalar:
                o << "Scalar";
                break;
            case OperationType::InnerProduct:
                o << "InnerProduct";
                break;
            case OperationType::Convolution:
                o << "Convolution";
                break;
            case OperationType::MaxPool:
                o << "MaxPool";
                break;
            case OperationType::Concat:
                o << "Concat";
                break;

        }
        return o;
    }


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
    __global__ void doConcatInterleave(float** targets, int numTargets, float* result, int batchSize) 
    {
        int elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
        int targetIdx = blockIdx.y * blockDim.y + threadIdx.y;
        int batchIdx = blockIdx.z * blockDim.z + threadIdx.z;

        if(targetIdx < numTargets 
                && batchIdx < batchSize 
                && elemIdx < concatSizes[targetIdx] // hmm, not very cuda kernel friendly 

          ) {
            int idx = elemIdx + concatSizes[targetIdx] * targetIdx + batchIdx * numTargets;

            result[idx] = targets[targetIdx][elemIdx + batchIdx * concatSizes[targetIdx]];

        }
    }
    __global__ void doConcatDeinterleave(float* inputSeed, int numTargets, float* result, int batchSize) {
        int elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
        int targetIdx = blockIdx.y * blockDim.y + threadIdx.y;
        int batchIdx = blockIdx.z * blockDim.z + threadIdx.z;

        int inputIdx = elemIdx + concatSizes[targetIdx] * targetIdx + batchIdx * numTargets;

        int pastCells = 0;
        // very unfriendky to cuda kernels...
        for(int i =0; i<targetIdx;i++)
            pastCells += concatSizes[targetIdx] * batchSize;


        int resultIdx = elemIdx + batchIdx * concatSizes[targetIdx] + targetIdx * pastCells;
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
        int srcSize = (rows+rowPadding)*(cols+colPadding);
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

    Operation Operation::applyLeakyReLU(string name, string target, uint rows, uint cols) {
        BasicConfig config = { .target{target}};
        return { .opType=OperationType::LeakyReLU
                , .workingSize = rows*cols
                , .resultSize = rows*cols
                , .gradSize = rows*cols
                , .rows=rows
                , .cols=cols
                , .name{name}
                , .noOp = false 
                , .config{ (BasicConfig){ .target{target} } }
                };
    }

    Operation Operation::innerProduct(string name, string target1, string target2, uint rows) {
        return { .opType=OperationType::InnerProduct
               , .workingSize = 0
               , .resultSize = 1
               , .gradSize = 2*rows
               , .rows = 1
               , .cols = 1
               , .name{name}
               , .noOp = false
               , .config{ (BinaryOpConfig) {.target1{target1}, .target2{target2}, 
                                            .targetRows = rows, .targetCols = 1}}
        };

    }

    Operation Operation::add(string name, string target1, string target2, uint rows, uint cols) {
        return { .opType=OperationType::Add
               , .workingSize = 0
               , .resultSize = rows*cols
               , .gradSize = rows*cols
               , .rows = rows
               , .cols = cols
               , .name{name}
               , .noOp = false
               , .config{ (BinaryOpConfig) {.target1{target1}, .target2{target2}, 
                                            .targetRows = rows, .targetCols = 1}}
        };
    }

    Operation Operation::scalarMultiply(string name, string target, uint rows, uint cols, float scale) {
        return { .opType=OperationType::Scalar
               , .workingSize = 0
               , .resultSize = rows*cols
               , .gradSize = rows*cols
               , .rows = rows
               , .cols = cols
               , .name{name}
               , .noOp = false
               , .config{ (ScalarConfig) { .target{target}, .scale=scale }}
        };
    }
    
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
               , .noOp = false
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
                , .noOp = false
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

    // // so we are assuming the results of each of the targets are just a big 
    // // continuous block in memory.  otherwise this won't work.  so be careful
    // // using it.  it's mostly used to push the gradients down.
    // UPDATE: no longer to a big continuous block of memory.  now we do copying
    // and interleaving, which is slower.  concat adds a step.
    Operation Operation::concat(string name, const vector<string>& targets, uint size) {
        return { .opType=OperationType::Concat
               , .workingSize = 0
               , .resultSize = size // we assume the user set this correctly.
               , .gradSize = 0
               , .rows = size 
               , .cols = 1
               , .name{name}
               , .noOp = false
               , .config { (ConcatConfig) {
                     .targets{targets}
                   , .size=size
               }}
        };
    }

    Function::Function(cublasHandle_t* cublasH) : ops(), memLocs(), cublasH(cublasH) {
        batchSize = 1; 
    }

    void Function::compile(uint _batchSize) {
        for(auto& op: ops) {
            opsMap[op.name] = &op;
        }

        this->batchSize = _batchSize;
        totalSize = 0; gradSize = 0; workingSize = 0; resultSize = 0;
        for(auto& op : ops){
            int batchSizeMultiplier = batchSize;
            if( op.opType == OperationType::WeightsMatrix ) 
                batchSizeMultiplier = 1;

            if( op.opType == OperationType::Concat ) {
                ConcatConfig opCnfg = get<ConcatConfig>(op.config);
                if(opCnfg.targets.size() == 1)
                    continue;
                int targetResults = 0;
                for(auto targetName:opCnfg.targets){ 
                    if(auto it = opsMap.find(targetName); it != opsMap.end()) {
                      Operation* target = it->second;
                      targetResults += target->resultSize * _batchSize;
                    }
                }
                op.resultSize = targetResults;
                op.gradSize = targetResults;
                op.workingSize = 0;
                
            } 
            totalSize += (op.workingSize + op.resultSize + op.gradSize) * batchSizeMultiplier;
            gradSize += op.gradSize * batchSizeMultiplier;
            workingSize += op.workingSize * batchSizeMultiplier;
            resultSize += op.resultSize * batchSizeMultiplier;
            
        }
        cudaMalloc((void**)&d_value,  totalSize  * sizeof(float));
        cudaMemset(d_value, 0, totalSize * sizeof(float));
        
        int idxGradSize = 0, idxWorkingSize = 0, idxResultSize = 0;
        for(auto op:ops){
            int batchSizeMultiplier = batchSize;
            if( op.opType == OperationType::WeightsMatrix )
                batchSizeMultiplier = 1;
            //if(op.opType == OperationType::Concat) {
                // the below line used to work before batch processing, it was sort
                // of a messy hack and relied on the user adding operations with
                // the targets of the concat operation coming all in a sequence together.
                // so we'd have A B in memory. but with batch processing we expect
                // A_1 ... A_n B1 ... B_n, so this line doesn't work.
                // So we give Concat a non-zero size and proceed as usual.
                // a smarter compile would compute the sum of the size of the targets.
                // but for now we just assume the user set that correction.
                // memLocs[op.name+"_result"] = memLocs[opCnfg.targets[0]+"_result"];
                //
            //}
            memLocs[op.name+"_grad"] = d_value + idxGradSize;
            idxGradSize += op.gradSize * batchSizeMultiplier;
            memLocs[op.name+"_result"] = d_value + gradSize + idxResultSize;
            idxResultSize += op.resultSize * batchSizeMultiplier;
            memLocs[op.name+"_working"] = d_value + gradSize + resultSize + idxWorkingSize;
            idxWorkingSize += op.workingSize * batchSizeMultiplier;

        }
    }

    void Function::resetGrad() {
        cudaErrCk( cudaMemset(d_value, 0, gradSize * sizeof(float)) );
        cudaErrCk( cudaMemset(d_value+gradSize+resultSize, 0, workingSize * sizeof(float)) );
        
    }

    Function::~Function() {
        cudaFree(d_value);
        
    }

    void Function::addOp(Operation op) {
        if(op.noOp) {
            Operation& lastOp = ops[ops.size()-1] ;
            Operation newOp { .opType = op.opType
                            , .workingSize = 0
                            , .resultSize = lastOp.resultSize
                            , .gradSize = lastOp.resultSize
                            , .rows = lastOp.rows
                            , .cols = lastOp.cols
                            , .name{op.name}
                            , .noOp = true 
                            , .config = lastOp.config
                            };
            ops.push_back(newOp);
        } else
            ops.push_back(op);
    }

    void Function::setValue(string name, const vector<vector<float>>& values) {
        const auto op = find_if(ops.begin(), ops.end(), [name](auto needle) { return needle.name == name;});
        if(op == ops.end()) {
          cout << "setValue cannot find op " << name << endl;
          exit(1);
          return;
        }
        if(values.size() != batchSize && op->opType != OperationType::WeightsMatrix) {
          cout << "setValues on " << name << " doesn't match the batchSize. batchSize is "
               << batchSize << " while the values size is " << values.size() << "." << endl;
          exit(1);
          return;
        }
        if(values[0].size() != op->resultSize) {
          cout << "setValues values passed in for " << name << " do not match op's result size. resultSize is "
               << op->resultSize << " while the values size is " << values[0].size() << "." << endl;
          exit(1);
          return;
        }
        // first we need to flatten the vector so we can have one nice contiguous memory block to copy to cuda
        // device. 
        // hmmmm...maybe vector<vector> isn't the right type...
        vector<float> total;
        for(const auto v : values) {
            total.insert(total.end(), v.begin(), v.end());
        }
        cudaMemcpy(memLocs[name+"_result"], &(total[0]), sizeof(float)*total.size(), cudaMemcpyHostToDevice);
    }

    void Function::gradDescent(string name, float learningRate) {
        float alpha = 1;
        float effectiveLR = learningRate * sqrt(batchSize);  
        float beta = -effectiveLR / batchSize;
        //float beta = -1 * learningRate / batchSize;
        Operation *op;

        for(auto needle : ops){
            if(needle.name == name) {
                op = &needle;
                break;
            }
        }
        if(op->opType != OperationType::WeightsMatrix) return;

        float* result  = memLocs[name+"_result"];

        for(int i=0;i<batchSize;i++){
          float* grad  = memLocs[name+"_grad"] + i * op->rows * op->cols;
          cublasErrCk ( cublasSgeam( *cublasH
                       , CUBLAS_OP_N
                       , CUBLAS_OP_N
                       , op->cols, op->rows, &alpha
                       , result, op->cols, &beta
                       , grad, op->cols, result, op->cols) );
        }

    }

    void Function::getValue(string name, vector<vector<float>>* results) {
        const auto op = find_if(ops.begin(), ops.end(), [name](auto needle) { return needle.name == name;});
        if(op == ops.end()) {
          cout << "getValue cannot find op " << name << endl;
          exit(1);
          return;
        }
        float* resultsTemp = new float [ batchSize * op->rows * op-> cols ];
        float* d_value = memLocs[name+"_result"];

        cudaMemcpy(resultsTemp, d_value, sizeof(float)*batchSize*op->rows*op->cols,cudaMemcpyDeviceToHost);

        for(int i=0;i<batchSize;i++) {
            results->push_back(vector(resultsTemp + i * op->resultSize, resultsTemp + ((i+1) * op->resultSize) ));
        }
        delete [] resultsTemp ;
    }

    void Function::getGrad(string name, vector<vector<float>>* results) {
        Operation *op;
        for(auto needle : ops){
            if(needle.name == name) {
                op = &needle;
                break;
            }
        }
        float* resultsTemp = new float [ batchSize * op->gradSize];
        float* d_value = memLocs[name+"_grad"];
        int batchSizeOverride = batchSize;
        if(op->opType == OperationType::WeightsMatrix)
          batchSizeOverride = 1;

        cudaMemcpy(resultsTemp, d_value, sizeof(float)*op->gradSize*batchSizeOverride,cudaMemcpyDeviceToHost);

        for(int i=0;i<batchSizeOverride;i++)
            results->push_back(vector(resultsTemp + i * op->gradSize, resultsTemp + ((i+1)*op->gradSize)));

        delete [] resultsTemp;

    }

    void Function::computeGrad(string name) {
        const auto op = find_if(ops.begin(), ops.end(), [name](auto needle) { return needle.name == name;});
        if(op == ops.end()) {
          cout << "computeGrad cannot find op " << name << endl;
          exit(1);
          return;
        }
        float* seed;
        cudaError_t err;
        int size = op->rows * op->cols * sizeof(float);
        err = cudaMalloc((void**)&seed, size*batchSize);
        if (err != cudaSuccess) {
            printf("malloc error in Function::computeGrad: %s\n", cudaGetErrorString(err));
            exit(1);
        }

        dim3 gd(ceil((op->cols * batchSize )/ 32.0), ceil(op->rows / 32.0), 1);
        dim3 bd(32, 32, 1);
        doFill<<<gd, bd>>>(op->rows*batchSize, op->cols, 1.0f, seed);

        computeGrad(name, seed);
        cudaFree(seed);
        

    }

    void Function::computeGrad(string name, float* seed){
        const auto it = find_if(ops.begin(), ops.end(), [name](auto needle) { return needle.name == name;});
        if(it == ops.end()) {
          cout << "computeGrad* cannot find op " << name << endl;
          exit(1);
          return;
        }

        Operation op = *it;
        switch(op.opType) {
            case OperationType::InnerProduct: {
                BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                float* col1 = memLocs[opConfig.target1+"_result"];
                float* col2 = memLocs[opConfig.target2+"_result"];

                float* vec1 = memLocs[name+"_grad"];
                float* vec2 = memLocs[name+"_grad"] + opConfig.targetRows * batchSize;
                
                cublasErrCk( cublasSetPointerMode( *cublasH, CUBLAS_POINTER_MODE_DEVICE) );

                cublasErrCk( cublasScopy(*cublasH, opConfig.targetRows * batchSize, col1, 1, vec1, 1) );
                cublasErrCk( cublasSscal(*cublasH, opConfig.targetRows * batchSize, seed, vec1, 1) );
                
                cublasErrCk( cublasScopy(*cublasH, opConfig.targetRows * batchSize, col2, 1, vec2, 1) );
                cublasErrCk( cublasSscal(*cublasH, opConfig.targetRows * batchSize, seed, vec2, 1) );
                cublasErrCk( cublasSetPointerMode( *cublasH, CUBLAS_POINTER_MODE_HOST ) );

                
                computeGrad(opConfig.target2, vec1);
                computeGrad(opConfig.target1, vec2);
            } break;
            case OperationType::InputColumn: {
                float alpha = 1;
                float *grad = memLocs[name+"_grad"];
                cublasErrCk( cublasSaxpy(*cublasH, op.gradSize*batchSize, &alpha, seed, 1, grad, 1) );

            } break;
            case OperationType::InputMatrix: {
                float alpha = 1;
                float *grad = memLocs[name+"_grad"];
                cublasErrCk( cublasSaxpy(*cublasH, op.gradSize*batchSize, &alpha, seed, 1, grad, 1) );
            } break;
            case OperationType::WeightsMatrix: {
                float alpha = 1;
                float *grad = memLocs[name+"_grad"];
                // we don't batch weights, because that's silly.  so we are summing all these up
                for(int i = 0; i <batchSize; i++)
                  cublasErrCk( cublasSaxpy(*cublasH, op.gradSize, &alpha, seed+i*op.gradSize, 1, grad, 1) );
            } break;
            case OperationType::MatrixProduct: {
                BinaryMatrixConfig opConfig = get<BinaryMatrixConfig>(op.config);

                const auto targetOp1 = find_if(ops.begin(), ops.end()
                            , [opConfig](auto needle) { return needle.name == opConfig.target1;});
                const auto targetOp2 = find_if(ops.begin(), ops.end()
                            , [opConfig](auto needle) { return needle.name == opConfig.target2;});
                if(targetOp1 == ops.end() || targetOp2 == ops.end()) {
                  cout << "compute(MatrixProduct) cannot find op " <<
                          opConfig.target1 << " or " << opConfig.target2 << endl;

                  exit(1);
                  return;
                }

                float *colValue = memLocs[opConfig.target2+"_result"];
                float *matrixGrad = memLocs[op.name+"_grad"];
                float *matrixValue = memLocs[opConfig.target1+"_result"];
                float *colGrad = matrixGrad + batchSize * opConfig.target1Rows * opConfig.target1Cols;

                float *colValues[batchSize];
                float *matrixValues[batchSize];
                float *seeds[batchSize];
                float *matrixGrads[batchSize];
                float *colGrads[batchSize];

                for(int i=0;i<batchSize;i++) {
                  if(targetOp1->opType == OperationType::WeightsMatrix)
                    matrixValues[i] = matrixValue;
                  else
                    matrixValues[i] = matrixValue + i * batchSize;
                    
                  colValues[i] = colValue + i * targetOp2->resultSize;
                  seeds[i] = seed + i * opConfig.target1Rows;
                  matrixGrads[i] = matrixGrad + i * opConfig.target1Rows * opConfig.target1Cols;
                  colGrads[i] = colGrad + i * opConfig.target1Cols * opConfig.target2Cols;
                }

                float** d_matrixValues;
                float** d_colValues;
                float** d_seeds;
                float** d_matrixGrads;
                float** d_colGrads;
                cudaMalloc((void**)&d_matrixValues, batchSize * sizeof(float*));
                cudaMalloc((void**)&d_colValues, batchSize * sizeof(float*));
                cudaMalloc((void**)&d_seeds, batchSize * sizeof(float*));
                cudaMalloc((void**)&d_matrixGrads, batchSize * sizeof(float*));
                cudaMalloc((void**)&d_colGrads, batchSize * sizeof(float*));

                cudaMemcpy(d_matrixValues, matrixValues, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                cudaMemcpy(d_colValues, colValues, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                cudaMemcpy(d_seeds, seeds, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                cudaMemcpy(d_matrixGrads, matrixGrads, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                cudaMemcpy(d_colGrads, colGrads, batchSize * sizeof(float*), cudaMemcpyHostToDevice);

                float alpha = 1;
                float beta = 0;

                cublasErrCk( cublasSgemmBatched( *cublasH
                           , CUBLAS_OP_T
                           , CUBLAS_OP_N
                           , opConfig.target1Cols
                           , opConfig.target1Rows
                           , 1
                           , &alpha
                           , d_colValues
                           , 1
                           , d_seeds
                           , 1
                           , &beta
                           , d_matrixGrads
                           , opConfig.target1Cols
                           , batchSize) ); 
                
                cublasErrCk( cublasSgemmBatched( *cublasH
                           , CUBLAS_OP_N
                           , CUBLAS_OP_T
                           , 1
                           , opConfig.target1Cols
                           , opConfig.target1Rows
                           , &alpha
                           , d_seeds
                           , 1
                           , d_matrixValues
                           , opConfig.target1Cols
                           , &beta
                           , d_colGrads
                           , 1
                           , batchSize) );

                cudaFree(d_matrixValues); 
                cudaFree(d_colValues); 
                cudaFree(d_seeds); 
                cudaFree(d_matrixGrads); 
                cudaFree(d_colGrads); 
                computeGrad(opConfig.target1, matrixGrad);
                computeGrad(opConfig.target2, colGrad);
            } break;
            case OperationType::Scalar: {
                ScalarConfig opConfig = get<ScalarConfig>(op.config);
                float *result = memLocs[op.name+"_result"];
                float *grad = memLocs[name+"_grad"];

                cublasErrCk( cublasScopy(*cublasH, op.cols * op.rows * batchSize, seed, 1, grad, 1) );
                cublasErrCk( cublasSscal(*cublasH, op.cols * op.rows * batchSize, &(opConfig.scale), grad, 1) );
                computeGrad(opConfig.target, grad);

            } break;
            case OperationType::Add: {
                BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                float* copySeed = memLocs[op.name+"_grad"];

                cudaMemcpy(copySeed, seed, batchSize * op.rows * op.cols * sizeof(float), cudaMemcpyDeviceToDevice);
                computeGrad(opConfig.target1, seed);
                computeGrad(opConfig.target2, copySeed);
            } break;
            case OperationType::LeakyReLU: {
                BasicConfig opConfig = get<BasicConfig>(op.config);
                float* newSeed = memLocs[op.name+"_working"];
                float *grad = memLocs[name+"_grad"];
                dim3 bd(16, 16, 4);
                dim3 gd(ceil(op.cols / 16.0), ceil(op.rows / 16.0), ceil(batchSize / 4.0));

                doComponentProduct<<<gd, bd>>>(op.rows, op.cols, grad, seed, newSeed, batchSize);

                computeGrad(opConfig.target, newSeed);

            } break;
            case OperationType::Convolution: {
                ConvolutionConfig opCnfg = get<ConvolutionConfig>(op.config);

                float alpha = 1;
                float beta = 0;

                int startInput = opCnfg.kernelRows * opCnfg.kernelCols * batchSize;
                int startMatrix = startInput 
                    + (opCnfg.multiplicandRows * opCnfg.multiplicandCols * batchSize );
                int startCol = startMatrix
                    + (opCnfg.unrKrnlRows * opCnfg.unrKrnlCols )*batchSize;

                float* rolledKernelMatrixGrad = memLocs[op.name+"_grad"];
                float* inputGrad = memLocs[op.name+"_grad"] + startInput;
                float* matrixGrad = memLocs[op.name+"_grad"] + startMatrix; 
                float* colGrad = memLocs[op.name+"_grad"] + startCol; 

                float* paddedInput = memLocs[op.name+"_working"];
                float* unrolledKernel = memLocs[op.name+"_working"]
                                    + opCnfg.paddedInputSize * batchSize;

                float* h_seed[batchSize];
                float* h_paddedInput[batchSize];
                float* h_matrixGrad[batchSize];
                int matrixGradSize = opCnfg.unrKrnlCols * opCnfg.unrKrnlRows;
                int paddedInputSize = opCnfg.unrKrnlCols;
                int seedSize = opCnfg.unrKrnlRows;

                for(int i =0;i<batchSize;i++) {
                  h_seed[i] = seed + i * seedSize;
                  h_paddedInput[i] = paddedInput + i * paddedInputSize;
                  h_matrixGrad[i] = matrixGrad + i * matrixGradSize;
                }
                float** d_seed;
                float** d_paddedInput;
                float** d_matrixGrad;
                cudaMalloc((void**)&d_seed, batchSize * sizeof(float*));
                cudaMalloc((void**)&d_paddedInput, batchSize * sizeof(float*));
                cudaMalloc((void**)&d_matrixGrad, batchSize * sizeof(float*));
                
                cudaMemcpy(d_seed, h_seed, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                cudaMemcpy(d_paddedInput, h_paddedInput, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                cudaMemcpy(d_matrixGrad, h_matrixGrad, batchSize * sizeof(float*), cudaMemcpyHostToDevice);

                cublasErrCk( cublasSgemmBatched( *cublasH
                            , CUBLAS_OP_T
                            , CUBLAS_OP_N
                            , opCnfg.unrKrnlCols
                            , opCnfg.unrKrnlRows
                            , 1
                            , &alpha
                            , d_paddedInput
                            , 1
                            , d_seed
                            , 1
                            , &beta
                            , d_matrixGrad
                            , opCnfg.unrKrnlCols
                            , batchSize));
 
                dim3 bd(32, 1, 32);
                dim3 gd(ceil(opCnfg.unrKrnlCols / 32.0), 1, ceil(batchSize / 32.0));

                    
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
                    , batchSize);
                cudaErrCk( cudaPeekAtLastError() );


                cudaFree(d_paddedInput);
                cudaFree(d_matrixGrad);

                computeGrad(opCnfg.kernel, rolledKernelMatrixGrad);

                float* h_unrolledKernel[batchSize];
                float* h_colGrad[batchSize];
                for(int i =0;i<batchSize;i++) {
                  h_unrolledKernel[i] = unrolledKernel;
                  h_colGrad[i] = colGrad + i * opCnfg.unrKrnlCols;
                }
                float** d_unrolledKernel;
                float** d_colGrad;
                cudaMalloc((void**)&d_unrolledKernel, batchSize * sizeof(float*));
                cudaMalloc((void**)&d_colGrad, batchSize * sizeof(float*));
                cudaMemcpy(d_unrolledKernel, h_unrolledKernel, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                cudaMemcpy(d_colGrad, h_colGrad, batchSize * sizeof(float*), cudaMemcpyHostToDevice);

                cublasErrCk(
                  cublasSgemmBatched( *cublasH
                      , CUBLAS_OP_N
                      , CUBLAS_OP_T
                      , 1
                      , opCnfg.unrKrnlCols
                      , opCnfg.unrKrnlRows
                      , &alpha
                      , d_seed
                      , 1
                      , d_unrolledKernel
                      , opCnfg.unrKrnlCols
                      , &beta
                      , d_colGrad
                      , 1
                      , batchSize)
                );

                cudaFree(d_seed);
                cudaFree(d_unrolledKernel);
                cudaFree(d_colGrad);

                dim3 gd2( ceil(opCnfg.multiplicandCols / 16.0)
                        , ceil(opCnfg.multiplicandRows / 16.0)
                        , ceil(batchSize / 4.0));
                dim3 bd2(16, 16, 4);
                doCopyWithoutPadding<<<gd2, bd2>>>( colGrad
                        , inputGrad
                        , opCnfg.multiplicandRows
                        , opCnfg.multiplicandCols
                        , opCnfg.rowPadding
                        , opCnfg.colPadding
                        , batchSize);
                cudaErrCk( cudaPeekAtLastError() );
                computeGrad(opCnfg.multiplicand, inputGrad);
            }break;
            case OperationType::MaxPool: {
                MaxPoolConfig opConfig = get<MaxPoolConfig>(op.config);
                float* targetValue = memLocs[opConfig.target+"_result"];
                float* result = memLocs[op.name+"_result"];
                float* grad = memLocs[op.name+"_grad"];

                dim3 bd(16, 16, 4);
                dim3 gd(ceil(op.cols / 16.0), ceil(op.rows / 16.0), ceil(batchSize / 4.0));
                doMaxPoolGrad<<<gd, bd>>>(op.rows, op.cols, opConfig.targetRows, opConfig.targetCols, 
                    opConfig.rowSkip, opConfig.height, opConfig.colSkip, opConfig.width, 
                    targetValue, result, seed, grad, batchSize);
                cudaErrCk( cudaPeekAtLastError() );

                computeGrad(opConfig.target, grad);
            }break;
            case OperationType::Concat: {
                ConcatConfig opCnfg = get<ConcatConfig>(op.config);
                int targetSizes[32];
                int d_size = opCnfg.targets.size();
                int memIndex = 0;
                float* d_newSeed = memLocs[op.name+"_grad"];

                for (int i=0; i < d_size; i++) {
                    const auto targetOp = 
                        find_if( ops.begin()
                               , ops.end()
                               , [opCnfg,i](auto needle) {
                                       return needle.name == opCnfg.targets[i];
                                 });
                    if(targetOp == ops.end()) break;
                    targetSizes[i] = targetOp->resultSize;
                }

                 
                cudaMemcpyToSymbol(concatSizes, targetSizes, d_size * sizeof(int));

                dim3 bd(16,16,4);
                dim3 gd(ceil((float)targetSizes[0] / 16.0), ceil((float)d_size / 16.0), ceil(batchSize / 4.0));


                doConcatDeinterleave<<<gd, bd>>>(seed, d_size, d_newSeed, batchSize);
                cudaErrCk( cudaPeekAtLastError() );


                for (auto target : opCnfg.targets) {
                  const auto targetOp = 
                      find_if( ops.begin()
                             , ops.end()
                             , [target](auto needle) {
                                     return needle.name == target;
                               });
                  if(targetOp == ops.end()) break;
                  computeGrad(target, d_newSeed + memIndex);
                  memIndex += targetOp->rows * targetOp->cols;
                }



            }break;

        }
    }


    void Function::compute() {
        for(auto op : ops) {
            switch(op.opType) {
                case OperationType::InputColumn:
                    // basically noop
                break;
                case OperationType::InputMatrix:
                    // this is the same as InputColumn.  InputColumn is redundant
                case OperationType::WeightsMatrix:
                    // this is the same as InputColumn.  InputColumn is redundant
                break;
                case OperationType::Concat: {
                    // ~no operation here, we could do a copy to get all the targets~
                    // ~into one continuous block in memory, but we're just going~
                    // ~to assume that's the case. ~
                    // ~a smart compiler might enforce that, actually!~
                    // with batch processing this gets tricky. if we compute A and B
                    // then we store them as
                    // A_1 ... A_n B_1 ... B_n
                    // A smart compile could ensure that, but with batch processing
                    // we actually expect this tob e
                    // A_1 B_1 A_2 B_2 ... A_n B_n.
                    // so our operation here is actually just to do the interleaving.
                    // this is trivial is batchSize = 1.
                    ConcatConfig opConfig = get<ConcatConfig>(op.config);

                    vector<float*> hostTargets;
                    float** d_targets;
                    int targetSizes[32];
                    int d_size = opConfig.targets.size();
                    for(int i=0;i<opConfig.targets.size();i++) {
                        const auto targetOp = find_if(ops.begin(), ops.end()
                                    , [opConfig,i](auto needle) { return needle.name == opConfig.targets[i];});
                        if (targetOp == ops.end()) {
                            cout << "compute(Concat) cannot find op " <<
                              opConfig.targets[i];

                            exit(1);
                            return;
                        }
                        hostTargets.push_back(memLocs[opConfig.targets[i]+"_result"]);
                        targetSizes[i] = targetOp->resultSize;
                    }

                    cudaMemcpyToSymbol(concatSizes, targetSizes, d_size * sizeof(int));
                    cudaMalloc((void**)&d_targets, d_size * sizeof(float*));
                    cudaMemcpy(d_targets, &(hostTargets[0]), d_size*sizeof(float*), cudaMemcpyHostToDevice); 

                    dim3 bd(16,16,4);
                    dim3 gd(ceil((float)targetSizes[0] / 16.0), ceil((float)d_size / 16.0), ceil(batchSize / 4.0));


                    doConcatInterleave<<<gd, bd>>>(d_targets, d_size, memLocs[op.name+"_result"], batchSize);
                    cudaErrCk( cudaPeekAtLastError() );
                    cudaFree( d_targets );


                break;}
                case OperationType::MatrixProduct: {
                    BinaryMatrixConfig opConfig = get<BinaryMatrixConfig>(op.config);

                    float alpha = 1;
                    float beta = 0;
                    float* d_matrix1 = memLocs[opConfig.target1+"_result"];
                    float* d_matrix2 = memLocs[opConfig.target2+"_result"];
                    float* d_result = memLocs[op.name+"_result"];

                    // if target1 is an Input it must not batch, all the pointers must point to the same matrixValue

                    // the colValue however should be batched, meaning we can pass it in directly all the time.
                    const auto targetOp1 = find_if(ops.begin(), ops.end()
                                , [opConfig](auto needle) { return needle.name == opConfig.target1;});
                    const auto targetOp2 = find_if(ops.begin(), ops.end()
                                , [opConfig](auto needle) { return needle.name == opConfig.target2;});
                    if(targetOp1 == ops.end() || targetOp2 == ops.end()) {
                      cout << "compute(MatrixProduct) cannot find op " <<
                              opConfig.target1 << " or " << opConfig.target2 << endl;

                      exit(1);
                      return;
                    }

                    float* As[batchSize];
                    float* Bs[batchSize];
                    float* Cs[batchSize];
                    vector<tuple<float*, float*, float*>> targets;
                    for(int i=0;i<batchSize;i++){
                      if(targetOp1->opType == OperationType::InputMatrix || targetOp1->opType == OperationType::WeightsMatrix){
                        As[i] = d_matrix1;
                      } else {
                        As[i] = d_matrix1 + i * targetOp1->resultSize;
                      }
                      Bs[i] = d_matrix2 + i * targetOp2->resultSize;
                      Cs[i] = d_result + i * op.resultSize;
                    }
                    
                    float** d_As;
                    float** d_Bs;
                    float** d_Cs;
                    cudaMalloc((void**)&d_As, batchSize * sizeof(float*));
                    cudaMalloc((void**)&d_Bs, batchSize * sizeof(float*));
                    cudaMalloc((void**)&d_Cs, batchSize * sizeof(float*));
                    
                    cudaMemcpy(d_As, As, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_Bs, Bs, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_Cs, Cs, batchSize * sizeof(float*), cudaMemcpyHostToDevice);


                    cublasErrCk( cublasSgemmBatched( *cublasH
                               , CUBLAS_OP_N
                               , CUBLAS_OP_N
                               , opConfig.target2Cols
                               , opConfig.target1Rows
                               , opConfig.target1Cols
                               , &alpha
                               , d_Bs
                               , opConfig.target2Cols
                               , d_As
                               , opConfig.target1Cols
                               , &beta
                               , d_Cs
                               , opConfig.target2Cols
                               , batchSize) ) 
                   cudaFree(d_As);
                   cudaFree(d_Bs);
                   cudaFree(d_Cs);


                break;}

                case OperationType::LeakyReLU:{ 
                    // TODO accessing memLocs like this is quite error prone.
                    // should probably get some accessor functions so I don't make mistakes
                    // the compile can't catch...
                            
                    BasicConfig opConfig = get<BasicConfig>(op.config);
                    float* d_col = memLocs[opConfig.target+"_result"];
                    float* d_grad = memLocs[op.name+"_grad"];
                    float* d_result = memLocs[op.name+"_result"];

                    dim3 bd(16, 16, 4);
                    dim3 gd(ceil(op.cols / 16.0), ceil(op.rows / 16.0), ceil(batchSize / 4.0));


                    doLeakyReLU<<<gd, bd>>>(op.rows, op.cols, d_grad, d_col, d_result, batchSize);
                    cudaErrCk( cudaPeekAtLastError() );


                break;}
                case OperationType::Add:{
                    BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                    float* d_v1 = memLocs[opConfig.target1+"_result"];
                    float* d_v2 = memLocs[opConfig.target2+"_result"];
                    float* d_result = memLocs[op.name+"_result"];
                    float alpha = 1.0;
                    
                    cublasErrCk( cublasScopy(*cublasH, op.rows * op.cols * batchSize, d_v1, 1, d_result, 1) );
                    cublasErrCk( cublasSaxpy(*cublasH, op.rows * op.cols * batchSize, &alpha, d_v2, 1, d_result, 1) );
                break;}
                case OperationType::Scalar:{
                    ScalarConfig opConfig = get<ScalarConfig>(op.config);
                    float* d_target = memLocs[opConfig.target+"_result"];
                    float* d_result = memLocs[op.name+"_result"];
                    cublasErrCk( cublasScopy(*cublasH, op.rows * op.cols * batchSize, d_target, 1, d_result, 1) );
                    cublasErrCk( cublasSscal(*cublasH, op.rows * op.cols * batchSize, &(opConfig.scale), d_result, 1) );
                break;}
                case OperationType::InnerProduct:{
                    BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                    float* d_v1 = memLocs[opConfig.target1+"_result"];
                    float* d_v2 = memLocs[opConfig.target2+"_result"];
                    float* d_result = memLocs[op.name+"_result"];

                    float alpha = 1;
                    float beta = 0;

                    float* v1Ptrs[batchSize];
                    float* v2Ptrs[batchSize];
                    float* resultPtrs[batchSize];

                    for(int i = 0; i < batchSize; i++) {
                        v1Ptrs[i] = d_v1 + i * opConfig.targetRows;
                        v2Ptrs[i] = d_v2 + i * opConfig.targetRows;
                        resultPtrs[i] = d_result + i * 1; 
                    }

                    float** d_v1Ptrs;
                    float** d_v2Ptrs;
                    float** d_resultPtrs;
                    cudaMalloc((void**)&d_v1Ptrs, batchSize * sizeof(float*));
                    cudaMalloc((void**)&d_v2Ptrs, batchSize * sizeof(float*));
                    cudaMalloc((void**)&d_resultPtrs, batchSize * sizeof(float*));

                    cudaMemcpy(d_v1Ptrs, v1Ptrs, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_v2Ptrs, v2Ptrs, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_resultPtrs, resultPtrs, batchSize * sizeof(float*), cudaMemcpyHostToDevice);

                    cublasErrCk( cublasSgemvBatched( *cublasH
                                       , CUBLAS_OP_N
                                       , 1
                                       , opConfig.targetRows
                                       , &alpha
                                       , d_v1Ptrs
                                       , 1
                                       , d_v2Ptrs
                                       , 1
                                       , &beta
                                       , d_resultPtrs
                                       , 1
                                       , batchSize) );

                    cudaFree(d_v1Ptrs);
                    cudaFree(d_v2Ptrs);
                    cudaFree(d_resultPtrs);

                break;}
                case OperationType::Convolution:{
                    // pad the input, which means a copy to wocrking, which is slow
                    ConvolutionConfig opCnfg = get<ConvolutionConfig>(op.config);
                    float* input = memLocs[opCnfg.multiplicand+"_result"];
                    float* paddedInput = memLocs[op.name+"_working"];
                    convolutionPadInput( opCnfg.multiplicandRows
                                       , opCnfg.multiplicandCols
                                       , opCnfg.rowPadding
                                       , opCnfg.colPadding
                                       , input
                                       , paddedInput
                                       , batchSize);

                    // unroll the kernel..another slow copy to working.
                    float* kernel = memLocs[opCnfg.kernel+"_result"];
                    float* unrolledKernel = memLocs[op.name+"_working"] + opCnfg.paddedInputSize * batchSize;
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
                    float* output = memLocs[op.name+"_result"];

                    float* As[batchSize];
                    float* xs[batchSize];
                    float* ys[batchSize];

                    for(int i=0;i<batchSize;i++){
                      xs[i] = paddedInput + i * opCnfg.paddedInputSize;
                      As[i] = unrolledKernel;
                      ys[i] = output + i * op.resultSize;
                    }
                    
                    float** d_As;
                    float** d_xs;
                    float** d_ys;
                    cudaMalloc((void**)&d_As, batchSize * sizeof(float*));
                    cudaMalloc((void**)&d_xs, batchSize * sizeof(float*));
                    cudaMalloc((void**)&d_ys, batchSize * sizeof(float*));
                    
                    cudaMemcpy(d_As, As, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_xs, xs, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_ys, ys, batchSize * sizeof(float*), cudaMemcpyHostToDevice);

                    cublasErrCk( cublasSgemvBatched( *cublasH
                                , CUBLAS_OP_T
                                , opCnfg.unrKrnlCols
                                , opCnfg.unrKrnlRows
                                , &alpha
                                , d_As
                                , opCnfg.unrKrnlCols
                                , d_xs
                                , 1
                                , &beta
                                , d_ys
                                , 1
                                , batchSize ) );
                    cudaFree (d_As);
                    cudaFree (d_xs);
                    cudaFree (d_ys);

                break;}
                case OperationType::MaxPool: {
                    MaxPoolConfig opConfig = get<MaxPoolConfig>(op.config);
                    float* d_target = memLocs[opConfig.target+"_result"];
                    float* d_result = memLocs[op.name+"_result"];

                    dim3 bd(16, 16, 4);
                    dim3 gd(ceil(op.cols / 16.0), ceil(op.rows / 16.0), ceil(batchSize / 4.0));

                    doMaxPool<<<gd, bd>>>(op.rows, op.cols, opConfig.targetRows, opConfig.targetCols,
                        opConfig.rowSkip, opConfig.height, opConfig.colSkip, opConfig.width,
                        d_target, d_result, batchSize);
                    cudaErrCk( cudaPeekAtLastError() );

                break;}
            }
        }

    }

    void Function::batchCompute( const map<string, vector<vector<float>>*>& results
                               , const vector<string> targets 
                               , const map<string, vector<vector<float>>>& inputs
                               , void (*batchFunction)(Function*, int, int )) {
        // save the original memLocs since we move these aroudn a lot. restore
        // them at the end.
        map<string, float*> originalMemLocs = memLocs;

        map<string, Operation*> targetOps;
        int totalTargetSize = 0;
        int batchSize = 0;
        int totalSize = 0;
        for(const auto target : targets) {
          const auto targetOp = find_if(ops.begin(), ops.end(), [target](auto needle) { return needle.name == target;});
          if(targetOp != ops.end()) {
            targetOps[target] = &(*targetOp);
          } else {
            cout << "batchCompute cannot find op " << target << endl;
            exit(1);
            return;
          }

        }


        map<string, float*> inputLocs;
        map<string, int> inputSizes;
        float* d_inputs;
        for( auto const& [varName, data] : inputs) {
            batchSize = data.size();
            totalSize += data.size() * data[0].size();
        }
        for ( const auto target : targets ){
            totalTargetSize += targetOps[target]->resultSize * batchSize;
        }
        totalSize += totalTargetSize; 
        cudaErrCk ( cudaMalloc((void**)&d_inputs,  totalSize  * sizeof(float)) ) ;
        totalSize = 0;

        for( auto const& [varName, data] : inputs) {
            inputLocs[varName] = d_inputs + totalSize;
            for(int i = 0; i < data.size(); i++) {
                cudaErrCk( 
                  cudaMemcpy( inputLocs[varName] + i * data[i].size()
                            , &(data[i][0])
                            , sizeof(float) * data[i].size()
                            , cudaMemcpyHostToDevice)
                );
            }
            inputSizes[varName] = data[0].size();
            totalSize += data.size() * data[0].size();
        }
        for(const auto target : targets) {
            inputLocs[target] = d_inputs + totalSize;
            totalSize += targetOps[target]->resultSize * batchSize;

        }

        for(int i = 0; i < batchSize; i++) {
            for( const auto target: targets) {
                memLocs[target + "_result"] = inputLocs[target] + i * targetOps[target]->resultSize ;
            }
            for (auto const& [varName, data] : inputs) {
                memLocs[varName + "_result"] = inputLocs[varName] + i * inputSizes[varName];
            }
          resetGrad();
          compute();
          (*batchFunction)(this, i, batchSize);
        }
        for (const auto target: targets) {
            vector<vector<float>>* rows = results.at(target);
            float* temp = new float [batchSize * targetOps[target]->resultSize ];
            cudaErrCk(
              cudaMemcpy( temp
                        , inputLocs[target]
                        , batchSize * targetOps[target]->resultSize * sizeof(float)
                        , cudaMemcpyDeviceToHost)
            );
            for(int i=0;i<batchSize;i++) { 
                rows->push_back(vector(temp + i * targetOps[target]->resultSize, temp + ((i+1)*targetOps[target]->resultSize)));
            }
            delete [] temp;
        }

        memLocs = originalMemLocs;
        cudaErrCk(
            cudaFree( d_inputs )
        );

    }


    void Function::batchCompute( const map<string, vector<vector<float>>*>& results
                               , const vector<string> targets 
                               , const map<string, vector<vector<float>>>& inputs) {
        batchCompute( results, targets, inputs, noOpBatchCompute);
    }








} // namespace DA
