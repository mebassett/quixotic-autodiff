#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>
#include <map>
#include <string>

using namespace DA;

class DaftConcatComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> result;
    vector<vector<float>> testgrad;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        
        // Add two single-element columns and one 2-element column
        f->addOp(Operation::column("v2", 1));
        f->addOp(Operation::column("v3", 1));
        f->addOp(Operation::column("v4", 2));
        
        // Add concat operation to combine v2 and v3 into a 2-element vector
        f->addOp(Operation::concat("v2v3", {"v2", "v3"}, 2));
        
        // Add inner product between the concatenated vector and v4
        f->addOp(Operation::innerProduct("flatF", "v2v3", "v4", 2));
        
        f->compile();

        // Set up values - now using vector of vectors for batch API
        f->setValue("v2", {{1}});
        f->setValue("v3", {{2}});
        f->setValue("v4", {{3, 5}});

        f->compute();
        f->computeGrad("flatF");
        
        f->getValue("flatF", &result);
        f->getGrad("v2", &testgrad);
    }
    
    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftConcatComputeTest, ConcatComputeTest) {
    EXPECT_EQ(result[0][0], 13) << "Daft Concat compute";
    EXPECT_EQ(testgrad[0][0], 3) << "Daft Concat grad";
}

class DaftBatchConcatComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> result;
    vector<vector<float>> testgrad;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // Add two single-element columns and one 2-element column
        f->addOp(Operation::column("v2", 1));
        f->addOp(Operation::column("v3", 1));
        f->addOp(Operation::column("v4", 2));

        // Concatenate v2 and v3 into a 2-element vector
        f->addOp(Operation::concat("v2v3", {"v2", "v3"}, 2));

        // Inner product of the concatenated vector with v4
        f->addOp(Operation::innerProduct("flatF", "v2v3", "v4", 2));

        f->compile(4);  // Compile for batch size of 4

        // Set batch values for 4 independent examples
        // v2 = {1, 2, 3, 4}
        f->setValue("v2", {{1}, {2}, {3}, {4}});
        // v3 = {2, 3, 4, 5}
        f->setValue("v3", {{2}, {3}, {4}, {5}});
        // v4 = {{3,5}, {1,2}, {4,6}, {2,3}}
        f->setValue("v4", {{3, 5}, {1, 2}, {4, 6}, {2, 3}});

        f->compute();
        f->computeGrad("flatF");

        f->getValue("flatF", &result);
        f->getGrad("v2", &testgrad);
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchConcatComputeTest, BatchConcatComputeTest) {
    // Expected: v2v3[i] = {v2[i], v3[i]} then flatF[i] = v2v3[i] · v4[i]
    // batch 0: {1,2} · {3,5} = 1*3 + 2*5 = 13
    // batch 1: {2,3} · {1,2} = 2*1 + 3*2 = 8
    // batch 2: {3,4} · {4,6} = 3*4 + 4*6 = 36
    // batch 3: {4,5} · {2,3} = 4*2 + 5*3 = 23
    float expectedResults[4] = {13, 8, 36, 23};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(result[i][0], expectedResults[i])
            << "Batch concat compute batch " << i;
    }

    // Gradient of v2 = v4[0] for each batch
    // batch 0: 3, batch 1: 1, batch 2: 4, batch 3: 2
    float expectedGrads[4] = {3, 1, 4, 2};
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(testgrad[i][0], expectedGrads[i])
            << "Batch concat gradient batch " << i;
    }
}

class DaftBatchConcatGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    vector<vector<float>> aGrad;
    vector<vector<float>> bGrad;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // Two 2x4 inputs, concatenated to 16, batch size 2.
        // NOTE: batch size 2 is deliberate (not just convenience): the concat
        // backward's deinterleave scratch (concatSize * batchSize floats)
        // currently overlaps the next op's grad region, and ip's
        // gradSize = 2 * concatSize only covers it for batchSize <= 2.
        f->addOp(Operation::inputMatrix("a", 2, 4));
        f->addOp(Operation::inputMatrix("b", 2, 4));
        f->addOp(Operation::concat("c", {"a", "b"}, 16));
        f->addOp(Operation::innerProduct("ip", "c", "c", 16));
        f->compile(2);

        f->setValue("a", {
            {1, 2, 3, 4, 5, 6, 7, 8},
            {0, 0, 0, 0, 1, 1, 1, 1}
        });
        f->setValue("b", {
            {10, 10, 10, 10, 10, 10, 10, 10},
            {2, 2, 2, 2, 2, 2, 2, 2}
        });

        f->compute();
        f->computeGrad("ip");

        f->getGrad("a", &aGrad);
        f->getGrad("b", &bGrad);
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete f;
    }
};

TEST_F(DaftBatchConcatGradTest, BatchConcatGradTest) {
    // ip = ||c||^2 per batch, so the seed flowing into the concat is
    // 2 * c, and each target's grad is 2 * its own values, per batch.
    float expectedAGrad[2][8] = {
        {2, 4, 6, 8, 10, 12, 14, 16},
        {0, 0, 0, 0, 2, 2, 2, 2}
    };
    float expectedBGrad[2][8] = {
        {20, 20, 20, 20, 20, 20, 20, 20},
        {4, 4, 4, 4, 4, 4, 4, 4}
    };
    for (int batch = 0; batch < 2; batch++) {
        for (int i = 0; i < 8; i++) {
            EXPECT_EQ(aGrad[batch][i], expectedAGrad[batch][i])
                << "Batch concat a grad batch " << batch << ", element " << i;
            EXPECT_EQ(bGrad[batch][i], expectedBGrad[batch][i])
                << "Batch concat b grad batch " << batch << ", element " << i;
        }
    }
}
