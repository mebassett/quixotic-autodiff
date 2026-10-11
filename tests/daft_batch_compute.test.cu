#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>
#include <map>
#include <string>

using namespace DA;

class DaftBatchComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    map<string, vector<vector<float>>*> results;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        f->addOp(Operation::column("a", 2));
        f->addOp(Operation::column("b", 2));
        f->addOp(Operation::innerProduct("result1","a","b",2));
        f->addOp(Operation::innerProduct("result2", "result1", "result1", 1));
        f->compile();

        results["result1"] = new vector<vector<float>>;
        results["result2"] = new vector<vector<float>>;

        map<string, vector<vector<float>>> inputs
         {{"a", { { 1, 2}, {0,0}, {3, 4}}}, {"b", { { 0, 1}, {9,9}, {1,0}}}};

        f->batchCompute(results, {"result1","result2"}, inputs); 
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete results["result1"];
        delete results["result2"];
        delete f;
    }
};

TEST_F(DaftBatchComputeTest, BatchComputeTest) {
    EXPECT_EQ((*results["result1"])[0][0], 2) << "result1-0";
    EXPECT_EQ((*results["result1"])[1][0], 0) << "result1-1";
    EXPECT_EQ((*results["result1"])[2][0], 3) << "result1-2";

    EXPECT_EQ((*results["result2"])[0][0], 4) << "result2-0";
    EXPECT_EQ((*results["result2"])[1][0], 0) << "result2-1";
    EXPECT_EQ((*results["result2"])[2][0], 9) << "result2-2";
}

class DaftBatchMatrixComputeTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    map<string, vector<vector<float>>*> results;

    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);

        // Add two 4x4 matrices and one 4x1 column vector
        f->addOp(Operation::weightsMatrix("A1", 4, 4));
        f->addOp(Operation::weightsMatrix("A2", 4, 4));
        f->addOp(Operation::column("x", 4));
        
        // Perform sequential matrix multiplications: A1*x, then A2*(A1*x)
        f->addOp(Operation::matrixProduct("result1", "A1", "x", 4, 4, 1));
        f->addOp(Operation::matrixProduct("result2", "A2", "result1", 4, 4, 1));
        f->compile();

        // Set fixed matrix values
        f->setValue("A1", {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}});  // Identity matrix
        f->setValue("A2", {{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0}});  // Cyclic permutation matrix

        results["result1"] = new vector<vector<float>>;
        results["result2"] = new vector<vector<float>>;

        // Create batch inputs with 3 different column vectors only
        map<string, vector<vector<float>>> inputs {
            {"x", {
                {1, 2, 3, 4},  // Simple sequence
                {1, 1, 1, 1},  // All ones
                {2, 3, 5, 7}   // Prime-like sequence
            }}
        };

        f->batchCompute(results, {"result1", "result2"}, inputs);
    }

    void TearDown() override {
        cublasDestroy(cublasH);
        delete results["result1"];
        delete results["result2"];
        delete f;
    }
};

TEST_F(DaftBatchMatrixComputeTest, BatchMatrixComputeTest) {
    // Test batch 0: Identity * [1,2,3,4] = [1,2,3,4]
    EXPECT_EQ((*results["result1"])[0][0], 1) << "A1*x batch 0, element 0";
    EXPECT_EQ((*results["result1"])[0][1], 2) << "A1*x batch 0, element 1";
    EXPECT_EQ((*results["result1"])[0][2], 3) << "A1*x batch 0, element 2";
    EXPECT_EQ((*results["result1"])[0][3], 4) << "A1*x batch 0, element 3";

    // Test batch 0: Permutation * [1,2,3,4] = [2,3,4,1]
    EXPECT_EQ((*results["result2"])[0][0], 2) << "A2*(A1*x) batch 0, element 0";
    EXPECT_EQ((*results["result2"])[0][1], 3) << "A2*(A1*x) batch 0, element 1";
    EXPECT_EQ((*results["result2"])[0][2], 4) << "A2*(A1*x) batch 0, element 2";
    EXPECT_EQ((*results["result2"])[0][3], 1) << "A2*(A1*x) batch 0, element 3";

    // Test batch 1: Identity * [1,1,1,1] = [1,1,1,1]
    EXPECT_EQ((*results["result1"])[1][0], 1) << "A1*x batch 1, element 0";
    EXPECT_EQ((*results["result1"])[1][1], 1) << "A1*x batch 1, element 1";
    EXPECT_EQ((*results["result1"])[1][2], 1) << "A1*x batch 1, element 2";
    EXPECT_EQ((*results["result1"])[1][3], 1) << "A1*x batch 1, element 3";

    // Test batch 1: Permutation * [1,1,1,1] = [1,1,1,1]
    EXPECT_EQ((*results["result2"])[1][0], 1) << "A2*(A1*x) batch 1, element 0";
    EXPECT_EQ((*results["result2"])[1][1], 1) << "A2*(A1*x) batch 1, element 1";
    EXPECT_EQ((*results["result2"])[1][2], 1) << "A2*(A1*x) batch 1, element 2";
    EXPECT_EQ((*results["result2"])[1][3], 1) << "A2*(A1*x) batch 1, element 3";

    // Test batch 2: Identity * [2,3,5,7] = [2,3,5,7]
    EXPECT_EQ((*results["result1"])[2][0], 2) << "A1*x batch 2, element 0";
    EXPECT_EQ((*results["result1"])[2][1], 3) << "A1*x batch 2, element 1";
    EXPECT_EQ((*results["result1"])[2][2], 5) << "A1*x batch 2, element 2";
    EXPECT_EQ((*results["result1"])[2][3], 7) << "A1*x batch 2, element 3";

    // Test batch 2: Permutation * [2,3,5,7] = [3,5,7,2]
    EXPECT_EQ((*results["result2"])[2][0], 3) << "A2*(A1*x) batch 2, element 0";
    EXPECT_EQ((*results["result2"])[2][1], 5) << "A2*(A1*x) batch 2, element 1";
    EXPECT_EQ((*results["result2"])[2][2], 7) << "A2*(A1*x) batch 2, element 2";
    EXPECT_EQ((*results["result2"])[2][3], 2) << "A2*(A1*x) batch 2, element 3";
}

// ---------------------------------------------------------------------------
// batchCompute chunking / result-gathering with batchSize > 1.
// These guard the core loop: data is processed in `batchSize`-sized chunks and
// every one of the `batchComputeSize` samples must be gathered into `results`.
// ---------------------------------------------------------------------------
class DaftBatchComputeChunkingTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    map<string, vector<vector<float>>*> results;

    // r_i = a_i . b_i  (inner product of two 2-vectors)
    void build(int batchSize) {
        f = new Function(&cublasH);
        f->addOp(Operation::column("a", 2));
        f->addOp(Operation::column("b", 2));
        f->addOp(Operation::innerProduct("r", "a", "b", 2));
        f->compile(batchSize);
        results["r"] = new vector<vector<float>>;
    }

    void TearDown() override {
        if (results.count("r")) { delete results["r"]; results.erase("r"); }
        delete f;
        cublasDestroy(cublasH);
    }

    void SetUp() override { cublasCreate(&cublasH); f = nullptr; }
};

TEST_F(DaftBatchComputeChunkingTest, GathersAllSamplesWithBatchSize2) {
    build(2);   // 4 samples -> 2 chunks
    map<string, vector<vector<float>>> inputs = {
        {"a", {{1, 2}, {3, 4}, {5, 6}, {7, 8}}},
        {"b", {{1, 0}, {0, 1}, {1, 1}, {2, 0}}}
    };
    f->batchCompute(results, {"r"}, inputs);
    auto& r = *results["r"];
    ASSERT_EQ(r.size(), 4u) << "all 4 samples must be gathered, not just batchSize";
    float expected[4] = {1, 4, 11, 14};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(r[i][0], expected[i], 1e-5f) << "r sample " << i;
}

TEST_F(DaftBatchComputeChunkingTest, SingleChunkWhenSizeEqualsBatchSize) {
    build(4);   // 4 samples -> exactly 1 chunk
    map<string, vector<vector<float>>> inputs = {
        {"a", {{1, 1}, {2, 2}, {3, 3}, {4, 4}}},
        {"b", {{1, 0}, {0, 1}, {1, 1}, {1, -1}}}
    };
    f->batchCompute(results, {"r"}, inputs);
    auto& r = *results["r"];
    ASSERT_EQ(r.size(), 4u);
    float expected[4] = {1, 2, 6, 0};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(r[i][0], expected[i], 1e-5f) << "r sample " << i;
}

TEST_F(DaftBatchComputeChunkingTest, SingleSample) {
    build(1);
    map<string, vector<vector<float>>> inputs = {
        {"a", {{2, 3}}},
        {"b", {{4, 5}}}
    };
    f->batchCompute(results, {"r"}, inputs);
    auto& r = *results["r"];
    ASSERT_EQ(r.size(), 1u);
    EXPECT_NEAR(r[0][0], 2*4 + 3*5, 1e-5f);   // 19
}

// File-scope no-op batchFunction (C function pointer, as batchCompute requires).
static int g_noopCalls = 0;
static void noopFn(Function* /*f*/, int /*idx*/, int /*length*/) { g_noopCalls++; }

TEST_F(DaftBatchComputeChunkingTest, NoopBatchFunctionStillGathersResults) {
    build(2);
    g_noopCalls = 0;
    map<string, vector<vector<float>>> inputs = {
        {"a", {{1, 0}, {0, 1}, {1, 1}, {2, 0}}},
        {"b", {{1, 0}, {0, 1}, {1, 0}, {0, 1}}}
    };
    f->batchCompute(results, {"r"}, inputs, noopFn);
    EXPECT_EQ(g_noopCalls, 2) << "no-op callback should still run once per chunk";
    auto& r = *results["r"];
    ASSERT_EQ(r.size(), 4u);
    float expected[4] = {1, 1, 1, 0};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(r[i][0], expected[i], 1e-5f) << "r sample " << i;
}
