#include "daft_autodiff.h"
#include "daft_innerproduct.h"
#include "daft_convolution.h"
#include "daft_inputs.h"
#include "daft_matrix_product.h"
#include "daft_leaky_relu.h"
#include "daft_add.h"
#include "daft_scalar.h"
#include "daft_maxpool.h"
#include "daft_concat.h"
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
              InnerProduct_computeGrad(this, seed, op);

                
            } break;

            case OperationType::InputColumn: {
              Input_computeGrad(this, seed, op);
            } break;
            case OperationType::InputMatrix: {
              Input_computeGrad(this, seed, op);
            } break;
            case OperationType::WeightsMatrix: {
              Weights_computeGrad(this, seed, op);
            } break;
            case OperationType::MatrixProduct: {
              MatrixProduct_computeGrad(this, seed, op);
            } break;
            case OperationType::Scalar: {
              Scalar_computeGrad(this, seed, op);
            } break;
            case OperationType::Add: {
              Add_computeGrad(this, seed, op);
            } break;
            case OperationType::LeakyReLU: {
              LeakyReLU_computeGrad(this, seed, op);
            } break;
            case OperationType::Convolution: {
              Convolution_computeGrad(this, seed, op);
            }break;
            case OperationType::MaxPool: {
              MaxPool_computeGrad(this, seed, op);
            }break;
            case OperationType::Concat: {
              Concat_computeGrad(this, seed, op);
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
                  Concat_compute(this, op);
                break;}
                case OperationType::MatrixProduct: {
                  MatrixProduct_compute(this, op);
                break;}

                case OperationType::LeakyReLU:{ 
                  LeakyReLU_compute(this, op);
                break;}
                case OperationType::Add:{
                  Add_compute(this, op);
                break;}
                case OperationType::Scalar:{
                  Scalar_compute(this, op);
                break;}
                case OperationType::InnerProduct:{
                  InnerProduct_compute(this, op);
                break;}
                case OperationType::Convolution:{
                  Convolution_compute(this, op);
                break;}
                case OperationType::MaxPool: {
                  MaxPool_compute(this, op);
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
