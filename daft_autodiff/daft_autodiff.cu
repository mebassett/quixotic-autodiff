#include "daft_autodiff.h"
#include "daft_kernels.cuh"
#include "daft_error.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

namespace DA {

void noOpBatchCompute(Function* f, int idx, int length) {}


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
               , .config { (ConcatConfig) {
                     .targets{targets}
                   , .size=size
               }}
        };
    }

    Function::Function(cublasHandle_t* cublasH) : ops(), memLocs(), pointerLocs(), cublasH(cublasH) {
        batchSize = 1; 
    }

    float* Function::getResult(string opName){
      try{ 
        const MemoryLocation m = memLocs.at(opName);
        return m.result;
      } catch(out_of_range&){
          cout << "Function::getResult cannot find op: " << opName << endl;
          exit(1);
          return nullptr;
      }
    }

    float* Function::getGrad(string opName){
      try{ 
        const MemoryLocation m = memLocs.at(opName);
        return m.grad;
      } catch(out_of_range&){
          cout << "Function::getGrad cannot find op: " << opName << endl;
          exit(1);
          return nullptr;
      }
    }

    float* Function::getWorkingMemory(string opName){
      try{ 
        const MemoryLocation m = memLocs.at(opName);
        return m.working;
      } catch(out_of_range&){
          cout << "Function::getWorkingMemory cannot find op: " << opName << endl;
          exit(1);
          return nullptr;
      }
    }

    void Function::compile(uint _batchSize) {
        for(auto& op: ops) {
            opsMap[op.name] = &op;
        }

        this->batchSize = _batchSize;
        totalSize = 0; gradSize = 0; workingSize = 0; resultSize = 0;
        int pointerSize = 0;
        for(auto& op : ops){
            int batchSizeMultiplier = batchSize;
            if( op.opType == OperationType::WeightsMatrix ) 
                batchSizeMultiplier = 1;

            if( op.opType == OperationType::Concat ) {
                ConcatConfig opCnfg = get<ConcatConfig>(op.config);
                int targetResults = 0;
                for(auto targetName:opCnfg.targets){ 
                    if(auto it = opsMap.find(targetName); it != opsMap.end()) {
                      Operation* target = it->second;
                      targetResults += target->resultSize;
                    }
                }
                op.resultSize = targetResults;
                op.gradSize = targetResults;
                op.workingSize = 0;
                pointerSize += opCnfg.targets.size();
            }
            totalSize += (op.workingSize + op.resultSize + op.gradSize) * batchSizeMultiplier;
            gradSize += op.gradSize * batchSizeMultiplier;
            workingSize += op.workingSize * batchSizeMultiplier;
            resultSize += op.resultSize * batchSizeMultiplier;
            
        }
        cudaMalloc((void**)&d_value, totalSize * sizeof(float));
        cudaMemset(d_value, 0, totalSize * sizeof(float));
        cudaMalloc((void**)&d_pointers, pointerSize * sizeof(float*));
        
        int idxGradSize = 0, idxWorkingSize = 0, idxResultSize = 0, idxPointerSize = 0;
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
            float* currentGrad = d_value + idxGradSize;
            idxGradSize += op.gradSize * batchSizeMultiplier;
            float* currentResult = d_value + gradSize + idxResultSize;
            idxResultSize += op.resultSize * batchSizeMultiplier;
            float* currentWorking  = d_value + gradSize + resultSize + idxWorkingSize;
            memLocs[op.name] = { currentWorking, currentGrad, currentResult};

            idxWorkingSize += op.workingSize * batchSizeMultiplier;
            if( op.opType == OperationType::Concat ) {
              ConcatConfig opCnfg = get<ConcatConfig>(op.config);
              pointerLocs[op.name] = d_pointers + idxPointerSize;
              idxPointerSize += opCnfg.targets.size();

            }

        }
    }

    void Function::resetGrad() {
        cudaErrCk( cudaMemset(d_value, 0, gradSize * sizeof(float)) );
        cudaErrCk( cudaMemset(d_value+gradSize+resultSize, 0, workingSize * sizeof(float)) );
        
    }

    Function::~Function() {
        cudaFree(d_value);
        cudaFree(d_pointers);
        
    }

    void Function::addOp(Operation op) {
      ops.push_back(op);
    }

    void Function::setValue(string name, const vector<vector<float>>& values) {
        Operation* op;
        try {
          op = opsMap.at(name);
        } catch (const out_of_range&) {
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
        cudaMemcpy(getResult(name), &(total[0]), sizeof(float)*total.size(), cudaMemcpyHostToDevice);
    }

    void Function::gradDescent(string name, float learningRate) {
        float alpha = 1;
        float effectiveLR = learningRate * sqrt(batchSize);  
        float beta = -effectiveLR / batchSize;
        //float beta = -1 * learningRate / batchSize;
        Operation *op;
        try {
          op = opsMap.at(name);
        } catch(const out_of_range&){
          cout << "gradDescent: can't find op: " << name;
          exit(1);
          return;
        }

        if(op->opType != OperationType::WeightsMatrix) return;

        float* result  = getResult(name);
        float* grad  = getGrad(name);
        cublasErrCk ( cublasSgeam( *cublasH
                 , CUBLAS_OP_N
                 , CUBLAS_OP_N
                 , op->cols, op->rows, &alpha
                 , result, op->cols, &beta
                 , grad, op->cols, result, op->cols) );

    }

    void Function::getValue(string name, vector<vector<float>>* results) {
        Operation* op;
        try {
          op = opsMap.at(name);
        } catch (const out_of_range&) {
          cout << "getValue cannot find op " << name << endl;
          exit(1);
          return;
        }
        float* resultsTemp = new float [ batchSize * op->rows * op-> cols ];
        float* d_value = getResult(name);

        cudaMemcpy(resultsTemp, d_value, sizeof(float)*batchSize*op->rows*op->cols,cudaMemcpyDeviceToHost);

        for(int i=0;i<batchSize;i++) {
            results->push_back(vector(resultsTemp + i * op->resultSize, resultsTemp + ((i+1) * op->resultSize) ));
        }
        delete [] resultsTemp ;
    }

    void Function::getGrad(string name, vector<vector<float>>* results) {
        Operation *op;
        try {
          op = opsMap.at(name);
        } catch(const out_of_range&){
          cout << "getGrad: can't find op: " << name;
          exit(1);
          return;
        }
        float* resultsTemp = new float [ batchSize * op->gradSize];
        float* d_value = getGrad(name);
        int batchSizeOverride = batchSize;
        if(op->opType == OperationType::WeightsMatrix)
          batchSizeOverride = 1;

        cudaMemcpy(resultsTemp, d_value, sizeof(float)*op->gradSize*batchSizeOverride,cudaMemcpyDeviceToHost);

        for(int i=0;i<batchSizeOverride;i++)
            results->push_back(vector(resultsTemp + i * op->gradSize, resultsTemp + ((i+1)*op->gradSize)));

        delete [] resultsTemp;

    }

    void Function::computeGrad(string name) {
        Operation* op;
        try {
          op = opsMap.at(name);
        } catch (const out_of_range&) {
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
        Operation* it;
        try {
          it = opsMap.at(name);
        } catch (const out_of_range&) {
          cout << "computeGrad* cannot find op " << name << endl;
          exit(1);
          return;
        }

        Operation& op = *it;
        switch(op.opType) {
            case OperationType::InnerProduct: {
                BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                float* col1 = getResult(opConfig.target1);//memLocs[opConfig.target1+"_result"];
                float* col2 = getResult(opConfig.target2);//memLocs[opConfig.target2+"_result"];

                float* vec1 = getGrad(name);//memLocs[name+"_grad"];
                float* vec2 = vec1 + opConfig.targetRows * batchSize;
                
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
                float *grad = getGrad(name);//memLocs[name+"_grad"];
                cublasErrCk( cublasSaxpy(*cublasH, op.gradSize*batchSize, &alpha, seed, 1, grad, 1) );

            } break;
            case OperationType::InputMatrix: {
                float alpha = 1;
                float *grad = getGrad(name);//memLocs[name+"_grad"];
                cublasErrCk( cublasSaxpy(*cublasH, op.gradSize*batchSize, &alpha, seed, 1, grad, 1) );
            } break;
            case OperationType::WeightsMatrix: {
                float alpha = 1;
                float *grad = getGrad(name);//memLocs[name+"_grad"];
                // we don't batch weights, because that's silly.  so we are summing all these up
                for(int i = 0; i <batchSize; i++)
                  cublasErrCk( cublasSaxpy(*cublasH, op.gradSize, &alpha, seed+i*op.gradSize, 1, grad, 1) );
            } break;
            case OperationType::MatrixProduct: {
                BinaryMatrixConfig opConfig = get<BinaryMatrixConfig>(op.config);

                Operation* targetOp1;
                Operation* targetOp2;
                try {
                  targetOp1 = opsMap.at(opConfig.target1);
                  targetOp2 = opsMap.at(opConfig.target2);
                } catch (const out_of_range&) {
                  cout << "compute(MatrixProduct) cannot find op " <<
                          opConfig.target1 << " or " << opConfig.target2 << endl;

                  exit(1);
                  return;
                }

                float *colValue = getResult(opConfig.target2);//memLocs[opConfig.target2+"_result"];
                float *matrixGrad = getGrad(name);//memLocs[op.name+"_grad"];
                float *matrixValue = getResult(opConfig.target1);//memLocs[opConfig.target1+"_result"];
                float *colGrad = matrixGrad + batchSize * opConfig.target1Rows * opConfig.target1Cols;

                int matrixValueStride = targetOp1->rows * targetOp1->cols;
                if(targetOp1->opType == OperationType::WeightsMatrix)
                  matrixValueStride = 0;


                float alpha = 1;
                float beta = 0;

                cublasErrCk( cublasSgemmStridedBatched( *cublasH
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
                           , batchSize) ); 
                
                cublasErrCk( cublasSgemmStridedBatched( *cublasH
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
                           , batchSize) );

                computeGrad(opConfig.target1, matrixGrad);
                computeGrad(opConfig.target2, colGrad);
            } break;
            case OperationType::Scalar: {
                ScalarConfig opConfig = get<ScalarConfig>(op.config);
                float *result = getResult(name);//memLocs[op.name+"_result"];
                float *grad = getGrad(name);//memLocs[name+"_grad"];

                cublasErrCk( cublasScopy(*cublasH, op.cols * op.rows * batchSize, seed, 1, grad, 1) );
                cublasErrCk( cublasSscal(*cublasH, op.cols * op.rows * batchSize, &(opConfig.scale), grad, 1) );
                computeGrad(opConfig.target, grad);

            } break;
            case OperationType::Add: {
                BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                float* copySeed = getGrad(name);//memLocs[op.name+"_grad"];

                cudaMemcpy(copySeed, seed, batchSize * op.rows * op.cols * sizeof(float), cudaMemcpyDeviceToDevice);
                computeGrad(opConfig.target1, seed);
                computeGrad(opConfig.target2, copySeed);
            } break;
            case OperationType::LeakyReLU: {
                BasicConfig opConfig = get<BasicConfig>(op.config);
                float* newSeed = getWorkingMemory(name);//memLocs[op.name+"_working"];
                float *grad = getGrad(name);//memLocs[name+"_grad"];
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

                float* rolledKernelMatrixGrad = getGrad(name);//memLocs[op.name+"_grad"];
                float* inputGrad = getGrad(name) + startInput;
                float* matrixGrad = getGrad(name) + startMatrix; 
                float* colGrad = getGrad(name) + startCol; 

                float* paddedInput = getWorkingMemory(name);//memLocs[op.name+"_working"];
                float* unrolledKernel = getWorkingMemory(name)//memLocs[op.name+"_working"]
                                    + opCnfg.paddedInputSize * batchSize;

                int matrixGradSize = opCnfg.unrKrnlCols * opCnfg.unrKrnlRows;
                int paddedInputSize = opCnfg.unrKrnlCols;
                int seedSize = opCnfg.unrKrnlRows;


                cublasErrCk( cublasSgemmStridedBatched( *cublasH
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



                computeGrad(opCnfg.kernel, rolledKernelMatrixGrad);


                cublasErrCk(
                  cublasSgemmStridedBatched( *cublasH
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
                      , batchSize)
                );


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
                float* targetValue = getResult(opConfig.target);//memLocs[opConfig.target+"_result"];
                float* result = getResult(name);//memLocs[op.name+"_result"];
                float* grad = getGrad(name);//memLocs[op.name+"_grad"];

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
                float* d_newSeed = getGrad(name);//memLocs[op.name+"_grad"];

                for (int i=0; i < d_size; i++) {
                    Operation* targetOp;
                    try {
                        targetOp = opsMap.at(opCnfg.targets[i]);
                    } catch (const out_of_range&) {
                        break;
                    }
                    targetSizes[i] = targetOp->resultSize;
                }

                 
                setConcatSizes(targetSizes, d_size);

                dim3 bd(16,16,4);
                dim3 gd(ceil((float)targetSizes[0] / 16.0), ceil((float)d_size / 16.0), ceil(batchSize / 4.0));


                doConcatDeinterleave<<<gd, bd>>>(seed, d_size, d_newSeed, opCnfg.size, batchSize);
                cudaErrCk( cudaPeekAtLastError() );


                for (auto target : opCnfg.targets) {
                  Operation* targetOp;
                  try {
                    targetOp = opsMap.at(target);
                  } catch (const out_of_range&) {
                    break;
                  }
                  computeGrad(target, d_newSeed + memIndex);
                  memIndex += targetOp->rows * targetOp->cols * batchSize;
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
                    float** d_targets = pointerLocs[op.name];
                    int targetSizes[32];
                    int d_size = opConfig.targets.size();
                    for(int i=0;i<opConfig.targets.size();i++) {
                        Operation* targetOp;
                        try {
                            targetOp = opsMap.at(opConfig.targets[i]);
                        } catch (const out_of_range&) {
                            cout << "compute(Concat) cannot find op " <<
                              opConfig.targets[i];

                            exit(1);
                            return;
                        }
                        hostTargets.push_back(getResult(opConfig.targets[i]));//memLocs[opConfig.targets[i]+"_result"]);
                        targetSizes[i] = targetOp->resultSize;
                    }

                    setConcatSizes(targetSizes, d_size);
                    cudaMemcpy(d_targets, &(hostTargets[0]), d_size*sizeof(float*), cudaMemcpyHostToDevice); 

                    dim3 bd(16,16,4);
                    dim3 gd(ceil((float)targetSizes[0] / 16.0), ceil((float)d_size / 16.0), ceil(batchSize / 4.0));


                    doConcatInterleave<<<gd, bd>>>(d_targets, d_size, getResult(op.name), opConfig.size, batchSize);
                    cudaErrCk( cudaPeekAtLastError() );


                break;}
                case OperationType::MatrixProduct: {
                    BinaryMatrixConfig opConfig = get<BinaryMatrixConfig>(op.config);

                    float alpha = 1;
                    float beta = 0;

                    // if target1 is an Input it must not batch, all the pointers must point to the same matrixValue

                    // the colValue however should be batched, meaning we can pass it in directly all the time.
                    Operation* targetOp1;
                    Operation* targetOp2;
                    try {
                      targetOp1 = opsMap.at(opConfig.target1);
                      targetOp2 = opsMap.at(opConfig.target2);
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

                    float* d_A = getResult(opConfig.target1);//memLocs[opConfig.target1+"_result"];
                    float* d_B = getResult(opConfig.target2);//memLocs[opConfig.target2+"_result"];
                    float* d_C = getResult(op.name);//memLocs[op.name+"_result"];


                    cublasErrCk( cublasSgemmStridedBatched( *cublasH
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
                               , batchSize) ) 


                break;}

                case OperationType::LeakyReLU:{ 
                    // TODO accessing memLocs like this is quite error prone.
                    // should probably get some accessor functions so I don't make mistakes
                    // the compile can't catch...
                            
                    BasicConfig opConfig = get<BasicConfig>(op.config);
                    float* d_col = getResult(opConfig.target);//memLocs[opConfig.target+"_result"];
                    float* d_grad = getGrad(op.name);//memLocs[op.name+"_grad"];
                    float* d_result = getResult(op.name);//memLocs[op.name+"_result"];

                    dim3 bd(16, 16, 4);
                    dim3 gd(ceil(op.cols / 16.0), ceil(op.rows / 16.0), ceil(batchSize / 4.0));


                    doLeakyReLU<<<gd, bd>>>(op.rows, op.cols, d_grad, d_col, d_result, batchSize);
                    cudaErrCk( cudaPeekAtLastError() );


                break;}
                case OperationType::Add:{
                    BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                    float* d_v1 = getResult(opConfig.target1);//memLocs[opConfig.target1+"_result"];
                    float* d_v2 = getResult(opConfig.target2);//memLocs[opConfig.target2+"_result"];
                    float* d_result = getResult(op.name);//memLocs[op.name+"_result"];
                    float alpha = 1.0;
                    
                    cublasErrCk( cublasScopy(*cublasH, op.rows * op.cols * batchSize, d_v1, 1, d_result, 1) );
                    cublasErrCk( cublasSaxpy(*cublasH, op.rows * op.cols * batchSize, &alpha, d_v2, 1, d_result, 1) );
                break;}
                case OperationType::Scalar:{
                    ScalarConfig opConfig = get<ScalarConfig>(op.config);
                    float* d_target = getResult(opConfig.target);//memLocs[opConfig.target+"_result"];
                    float* d_result = getResult(op.name);//memLocs[op.name+"_result"];
                    cublasErrCk( cublasScopy(*cublasH, op.rows * op.cols * batchSize, d_target, 1, d_result, 1) );
                    cublasErrCk( cublasSscal(*cublasH, op.rows * op.cols * batchSize, &(opConfig.scale), d_result, 1) );
                break;}
                case OperationType::InnerProduct:{
                    BinaryOpConfig opConfig = get<BinaryOpConfig>(op.config);
                    float* d_v1 = getResult(opConfig.target1);//memLocs[opConfig.target1+"_result"];
                    float* d_v2 = getResult(opConfig.target2);//memLocs[opConfig.target2+"_result"];
                    float* d_result = getResult(op.name);//memLocs[op.name+"_result"];

                    float alpha = 1;
                    float beta = 0;

                    cublasErrCk( cublasSgemvStridedBatched( *cublasH
                                       , CUBLAS_OP_N
                                       , 1
                                       , opConfig.targetRows
                                       , &alpha
                                       , d_v1
                                       , 1
                                       , opConfig.targetRows
                                       , d_v2
                                       , 1
                                       , opConfig.targetRows
                                       , &beta
                                       , d_result
                                       , 1
                                       , 1
                                       , batchSize) );


                break;}
                case OperationType::Convolution:{
                    // pad the input, which means a copy to wocrking, which is slow
                    ConvolutionConfig opCnfg = get<ConvolutionConfig>(op.config);
                    float* input = getResult(opCnfg.multiplicand);//memLocs[opCnfg.multiplicand+"_result"];
                    float* paddedInput = getWorkingMemory(op.name);//memLocs[op.name+"_working"];
                    convolutionPadInput( opCnfg.multiplicandRows
                                       , opCnfg.multiplicandCols
                                       , opCnfg.rowPadding
                                       , opCnfg.colPadding
                                       , input
                                       , paddedInput
                                       , batchSize);

                    // unroll the kernel..another slow copy to working.
                    float* kernel = getResult(opCnfg.kernel);//memLocs[opCnfg.kernel+"_result"];
                    float* unrolledKernel = getWorkingMemory(op.name) //memLocs[op.name+"_working"] 
                                            + opCnfg.paddedInputSize * batchSize;
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
                    float* output = getResult(op.name);//memLocs[op.name+"_result"];

                    cublasErrCk( cublasSgemvStridedBatched( *cublasH
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
                                , batchSize ) );

                break;}
                case OperationType::MaxPool: {
                    MaxPoolConfig opConfig = get<MaxPoolConfig>(op.config);
                    float* d_target = getResult(opConfig.target);//memLocs[opConfig.target+"_result"];
                    float* d_result = getResult(op.name);//memLocs[op.name+"_result"];

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
        map<string, MemoryLocation> originalMemLocs = memLocs;

        map<string, Operation*> targetOps;
        int totalTargetSize = 0;
        int batchComputeSize = 0;
        int totalSize = 0;
        for(const auto target : targets) {
          Operation* targetOp;
          try {
            targetOp = opsMap.at(target);
          } catch (const out_of_range&) {
            cout << "batchCompute cannot find op " << target << endl;
            exit(1);
            return;
          }
          targetOps[target] = targetOp;

        }


        map<string, float*> inputLocs;
        map<string, int> inputSizes;
        float* d_inputs;
        for( auto const& [varName, data] : inputs) {
            batchComputeSize = data.size();
            totalSize += data.size() * data[0].size();
        }
        if( batchComputeSize % batchSize != 0) {
            cout << "batchCompute's size must be a multiple of the compiled batchSize" << endl;
            exit(1);
            return;
        }

        for ( const auto target : targets ){
            totalTargetSize += targetOps[target]->resultSize * batchComputeSize;
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
            totalSize += targetOps[target]->resultSize * batchComputeSize;

        }


        for(int i = 0; i < batchComputeSize; i = i + batchSize ) {
            for( const auto target: targets) {
                MemoryLocation m = memLocs[target];
                memLocs[target] = { m.working, m.grad, inputLocs[target] + i * targetOps[target]->resultSize};
                 // inputLocs[target] + i * targetOps[target]->resultSize ;
            }
            for (auto const& [varName, data] : inputs) {
                MemoryLocation m = memLocs[varName];
                memLocs[varName] = { m.working, m.grad, inputLocs[varName] + i * inputSizes[varName] };
                 //inputLocs[varName] + i * inputSizes[varName];
            }
          resetGrad();
          compute();
          (*batchFunction)(this, i, batchComputeSize);
        }
        for (const auto target: targets) {
            vector<vector<float>>* rows = results.at(target);
            float* temp = new float [batchComputeSize * targetOps[target]->resultSize ];
            cudaErrCk(
              cudaMemcpy( temp
                        , inputLocs[target]
                        , batchComputeSize * targetOps[target]->resultSize * sizeof(float)
                        , cudaMemcpyDeviceToHost)
            );
            for(int i=0;i<batchComputeSize;i++) { 
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
