// daft_autodiff.h
#ifndef DAFT_AUTODIFF_H
#define DAFT_AUTODIFF_H
#include <cublas_v2.h>
#include <map>
#include <string>
#include <variant>
#include <vector>
#include <iostream>
#include <unordered_map>

using namespace std;

namespace DA {

enum class OperationType 
    { InputColumn
    , InputMatrix
    , WeightsMatrix
    , MatrixProduct
    , LeakyReLU
    , Add
    , Scalar
    , InnerProduct
    , Convolution
    , MaxPool
    , Concat
    };

ostream& operator<<(ostream &o, const OperationType t) ;

struct BasicConfig {
    string target;
};

struct ConcatConfig {
    vector<string> targets;
    const uint size;
};

struct BinaryOpConfig {
    string target1;
    string target2;
    const uint targetRows;
    const uint targetCols;
};

struct BinaryMatrixConfig {
    string target1;
    string target2;
    const uint target1Rows;
    const uint target1Cols;
    const uint target2Cols;
};

struct ConvolutionConfig {
    string multiplicand;
    string kernel;
    const uint rowPadding;
    const uint rowSkip;
    const uint colPadding;
    const uint colSkip;
    const uint multiplicandRows;
    const uint multiplicandCols;
    const uint kernelRows;
    const uint kernelCols;
    const uint unrKrnlRows;
    const uint unrKrnlCols;
    const uint paddedInputSize;
};

struct ScalarConfig {
    string target;
    float scale;
};

struct MaxPoolConfig {
    string target;
    const uint width;
    const uint height;
    const uint rowSkip;
    const uint colSkip;
    const uint targetRows;
    const uint targetCols;
};

using OpConfig = variant<BasicConfig, BinaryOpConfig, BinaryMatrixConfig, ScalarConfig, ConvolutionConfig, MaxPoolConfig, ConcatConfig>;

struct Operation {
    const OperationType opType;
    uint workingSize;
    uint resultSize;
    uint gradSize;
    const uint rows;
    const uint cols;
    const string name;
    const bool noOp = false;
    const OpConfig config;

    static Operation column(string name, uint rows);
    static Operation inputMatrix(string name, uint rows, uint cols);
    static Operation weightsMatrix(string name, uint rows, uint cols);
    static Operation matrixProduct(string name, string target1, string target2, uint target1Rows, uint target1Cols, uint target2Cols);
    static Operation applyLeakyReLU(string name, string target, uint rows, uint cols);
    static Operation innerProduct(string name, string target1, string target2, uint rows);
    static Operation add(string name, string target1, string target2, uint rows, uint cols);
    static Operation scalarMultiply(string name, string target, uint rows, uint cols, float scale);
    static Operation convolution(string name, string multiplicand, string kernel, uint rowPadding,
            uint rowSkip, uint colPadding, uint colSkip, 
            uint multiplicandRows, uint multiplicandCols,
            uint kernelRows, uint kernelCols);
    static Operation maxPool(string name, string target, uint width, uint height, 
            uint rowSkip, uint colSkip, uint targetRows, uint targetCols);
    static Operation concat(string name, const vector<string>& targets, uint size);
};


struct Function {
    vector<Operation> ops;
    unordered_map<string, Operation*> opsMap;
    uint gradSize;
    uint resultSize;
    uint workingSize;
    uint totalSize;
    uint batchSize;
    map<string, float*> memLocs;
    float* d_value;
    cublasHandle_t* cublasH;
    Function(cublasHandle_t* cublasH);
    void compile(uint _batchSize = 1);
    void resetGrad();
    void addOp(Operation); 
    void setValue(string name, const vector<vector<float>>& values);
    void compute();
    void gradDescent(string name, float learningRate);
    void getValue(string name, vector<vector<float>>* results);
    void getGrad(string name, vector<vector<float>>* result);
    void computeGrad(string name, float* seed);
    void computeGrad(string name);
    void batchCompute(const map<string, vector<vector<float>>*>& results, const vector<string> targets, const map<string, vector<vector<float>>>& inputs, void (*batchFunction)(Function*, int, int) );
    void batchCompute(const map<string, vector<vector<float>>*>& results, const vector<string> targets, const map<string, vector<vector<float>>>& inputs);
    ~Function();
};

}
#endif /* DAFT_AUTODIFF_H */
