#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;

class DaftMatrixColProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function *f;
    Function *g;
    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy", 2));
        f->addOp(Operation::weightsMatrix("abcd", 2, 2));
        f->addOp(Operation::matrixProduct("f", "abcd", "xy", 2, 2, 1));
        f->compile(); 

        g = new Function(&cublasH);
        g->addOp(Operation::weightsMatrix("A", 2, 2));
        g->addOp(Operation::inputMatrix("B", 2, 2));
        g->addOp(Operation::matrixProduct("g", "A", "B", 2, 2, 2));
        g-> compile();

    }
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
        delete g;
    }
};

TEST_F(DaftMatrixColProductTest, DaftMatrixColProductCompute) {
    f->setValue("abcd", {{1,-1,-1,1}});
    f->setValue("xy", {{1,2}});
    f->compute();
    f->computeGrad("f");
    vector<vector<float>> resultVec;
    vector<vector<float>> matrixGrad;
    f->getValue("f", &resultVec);
    f->getGrad("abcd", &matrixGrad);
    EXPECT_EQ(resultVec[0][0],-1) << "compute0";
    EXPECT_EQ(resultVec[0][1],1) << "compute1";
    EXPECT_EQ(matrixGrad[0][0], 1) << "abcd grad";
    EXPECT_EQ(matrixGrad[0][1], 2) << "abcd grad";
    EXPECT_EQ(matrixGrad[0][2], 1) << "abcd grad";
    EXPECT_EQ(matrixGrad[0][3], 2) << "abcd grad";

    g->setValue("A", {{1,2,3,4}});
    g->setValue("B", {{1,1,-1,1}});
    g->compute();
    vector<vector<float>> resultVec2;
    g->getValue("g", &resultVec2);
    EXPECT_EQ(resultVec2[0][0],-1) << "AB00";
    EXPECT_EQ(resultVec2[0][1],3) << "AB01";
    EXPECT_EQ(resultVec2[0][2],-1) << "AB10";
    EXPECT_EQ(resultVec2[0][3],7) << "AB11";
}

class DaftBatchMatrixColProductTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function *f;
    Function *g;

    void SetUp() override {
        cublasCreate(&cublasH);

        // f: fixed 2x2 matrix * batch of 2x1 columns -> 2x1 result
        f = new Function(&cublasH);
        f->addOp(Operation::column("xy", 2));
        f->addOp(Operation::weightsMatrix("abcd", 2, 2));
        f->addOp(Operation::matrixProduct("f", "abcd", "xy", 2, 2, 1));
        f->compile(3);  // Compile for batch size of 3

        // g: fixed 2x2 matrix * batch of 2x2 matrices -> 2x2 result
        g = new Function(&cublasH);
        g->addOp(Operation::weightsMatrix("A", 2, 2));
        g->addOp(Operation::inputMatrix("B", 2, 2));
        g->addOp(Operation::matrixProduct("g", "A", "B", 2, 2, 2));
        g->compile(2);  // Compile for batch size of 2
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
        delete g;
    }
};

TEST_F(DaftBatchMatrixColProductTest, BatchMatrixColProductCompute) {
    // f test: abcd = [[1,-1],[-1,1]] (fixed InputMatrix)
    f->setValue("abcd", {{1, -1, -1, 1}});
    f->setValue("xy", {
        {1, 2},     // batch 0
        {3, 4},     // batch 1
        {-1, 2}     // batch 2
    });
    f->compute();
    vector<vector<float>> resultVec;
    f->getValue("f", &resultVec);

    // batch 0: [[1,-1],[-1,1]] * [1,2] = [1*1+(-1)*2, (-1)*1+1*2] = [-1, 1]
    // batch 1: [[1,-1],[-1,1]] * [3,4] = [1*3+(-1)*4, (-1)*3+1*4] = [-1, 1]
    // batch 2: [[1,-1],[-1,1]] * [-1,2] = [(-1)+(-2), 1+2] = [-3, 3]
    float expected[3][2] = {
        {-1, 1},
        {-1, 1},
        {-3, 3}
    };
    for (int batch = 0; batch < 3; batch++) {
        for (int i = 0; i < 2; i++) {
            EXPECT_EQ(resultVec[batch][i], expected[batch][i])
                << "Batch matrix-col product batch " << batch << ", element " << i;
        }
    }

    // g test: A = [[1,2],[3,4]] (fixed InputMatrix)
    g->setValue("A", {{1, 2, 3, 4}});
    g->setValue("B", {
        {1, 1, -1, 1},   // batch 0
        {2, 0, 1, -1}    // batch 1
    });
    g->compute();
    vector<vector<float>> resultVec2;
    g->getValue("g", &resultVec2);

    // batch 0: A * [[1,1],[-1,1]] = [[1*1+2*(-1), 1*1+2*1], [3*1+4*(-1), 3*1+4*1]] = [[-1,3],[-1,7]]
    // batch 1: A * [[2,0],[1,-1]] = [[1*2+2*1, 1*0+2*(-1)], [3*2+4*1, 3*0+4*(-1)]] = [[4,-2],[10,-4]]
    float expected2[2][4] = {
        {-1, 3, -1, 7},
        {4, -2, 10, -4}
    };
    for (int batch = 0; batch < 2; batch++) {
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(resultVec2[batch][i], expected2[batch][i])
                << "Batch matrix-matrix product batch " << batch << ", element " << i;
        }
    }
}

class DaftBatchMatrixProductWeightsGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> weightGrad;
    vector<vector<float>> inputGrad;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // 2x3 weight matrix (shared across the batch) and a 3x1 input column
        f->addOp(Operation::weightsMatrix("W", 2, 3));
        f->addOp(Operation::column("x", 3));
        f->addOp(Operation::matrixProduct("prod", "W", "x", 2, 3, 1));
        f->compile(3);

        f->setValue("W", {{1, 2, 3, 4, 5, 6}});
        // x picks out each row of W, one per batch
        f->setValue("x", {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        });

        f->compute();
        f->computeGrad("prod");

        f->getGrad("W", &weightGrad);
        f->getGrad("x", &inputGrad);
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchMatrixProductWeightsGradTest, BatchMatrixProductWeightsGradTest) {
    // dL/dW = sum over batches of (seed * x_i^T) with seed = {1,1}:
    // {1,0;1,0} + {0,1;0,1} + {0,0;0,0} = all ones
    float expectedWeightGrad[6] = {1, 1, 1, 1, 1, 1};
    EXPECT_EQ(weightGrad.size(), 1u);
    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(weightGrad[0][i], expectedWeightGrad[i])
            << "Batch matrix product weight grad element " << i;
    }

    // dL/dx_i = W^T * {1,1} = {1+4, 2+5, 3+6}, same for every batch
    float expectedInputGrad[3] = {5, 7, 9};
    for (int batch = 0; batch < 3; batch++) {
        for (int i = 0; i < 3; i++) {
            EXPECT_EQ(inputGrad[batch][i], expectedInputGrad[i])
                << "Batch matrix product input grad batch " << batch << ", element " << i;
        }
    }
}

class DaftBatchMatrixProductMatrixTargetGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> aGrad;
    vector<vector<float>> xGrad;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // target1 is a per-batch InputMatrix here, not a shared weights matrix
        f->addOp(Operation::inputMatrix("A", 2, 3));
        f->addOp(Operation::column("x", 3));
        f->addOp(Operation::matrixProduct("prod", "A", "x", 2, 3, 1));
        f->compile(2);

        f->setValue("A", {
            {1, 1, 1, 1, 1, 1},   // batch 0: all ones
            {2, 2, 2, 2, 2, 2}    // batch 1: all twos
        });
        f->setValue("x", {
            {1, 0, 0},
            {0, 1, 0}
        });

        f->compute();
        f->computeGrad("prod");

        f->getGrad("A", &aGrad);
        f->getGrad("x", &xGrad);
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchMatrixProductMatrixTargetGradTest, BatchMatrixProductMatrixTargetGradTest) {
    // dL/dA_i = seed * x_i^T, seed = {1,1}
    float expectedAGrad[2][6] = {
        {1, 0, 0, 1, 0, 0},   // batch 0
        {0, 1, 0, 0, 1, 0}    // batch 1
    };
    for (int batch = 0; batch < 2; batch++) {
        for (int i = 0; i < 6; i++) {
            EXPECT_EQ(aGrad[batch][i], expectedAGrad[batch][i])
                << "Batch matrix product matrix-target grad batch " << batch << ", element " << i;
        }
    }

    // dL/dx_i = A_i^T * {1,1}
    float expectedXGrad[2][3] = {
        {2, 2, 2},
        {4, 4, 4}
    };
    for (int batch = 0; batch < 2; batch++) {
        for (int i = 0; i < 3; i++) {
            EXPECT_EQ(xGrad[batch][i], expectedXGrad[batch][i])
                << "Batch matrix product input grad batch " << batch << ", element " << i;
        }
    }
}
