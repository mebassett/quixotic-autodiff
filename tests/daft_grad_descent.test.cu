// Unit tests for Function::gradDescent, and for gradDescent being driven from
// a batchFunction inside Function::batchCompute.
//
// The network under test is the simplest gradient-bearing graph:
//     W (2x2 weights) , x (2x1 column) , prod = W * x  (2x1)
// For a (mini)batch of inputs x_0..x_{B-1} the summed weight gradient is
//     dL/dW[i][j] = sum_k x_k[j]      (seed is all-ones from computeGrad("prod"))
// so dL/dW = [[S0, S1], [S0, S1]] with Sj = sum_k x_k[j].
//
// gradDescent("W", lr) must apply the update exactly ONCE:
//     W_new = W + beta * dL/dW ,  beta = -lr * sqrt(batchSize) / batchSize
// (The pre-fix code applied it `batchSize` times and read out of bounds for
//  i > 0; these tests are written to catch that regression.)
#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <cstdio>

using namespace DA;

#define CUDA_TEST_CHECK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) fprintf(stderr, "CUDA error %s at %s:%d\n", \
        cudaGetErrorString(e_), __FILE__, __LINE__); } while (0)

// Read a shared (non-batched) weight buffer straight from device memory.
// getValue() is batch-oriented (it reads batchSize*rows*cols) and is therefore
// the wrong tool for a WeightsMatrix, whose buffer is only rows*cols.
static std::vector<float> readWeight(Function* f, const std::string& name, int n) {
    std::vector<float> h(n);
    CUDA_TEST_CHECK(cudaMemcpy(h.data(), f->memLocs[name + "_result"],
                               n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

static Function* makeProdNet(cublasHandle_t* h, int batchSize) {
    // NB: h is the *address* of a handle that outlives the Function (e.g. a
    // fixture member). Passing the handle by value and doing new Function(&h)
    // would leave the Function with a dangling cublasHandle_t*.
    // NB: input ops are added before the weights, matching the working examples.
    Function* f = new Function(h);
    f->addOp(Operation::column("x", 2));
    f->addOp(Operation::weightsMatrix("W", 2, 2));
    f->addOp(Operation::matrixProduct("prod", "W", "x", 2, 2, 1));
    f->compile(batchSize);
    return f;
}

// ---------------------------------------------------------------------------
// gradDescent, single sample (batchSize == 1)
// ---------------------------------------------------------------------------
class DaftGradDescentSingleTest : public testing::Test {
protected:
    cublasHandle_t h;
    Function* f;
    void SetUp() override {
        cublasCreate(&h);
        f = makeProdNet(&h, 1);
        f->setValue("W", {{1, 2, 3, 4}});
        f->setValue("x", {{1, 2}});
        f->compute();
        f->computeGrad("prod");
    }
    void TearDown() override { cublasDestroy(h); delete f; }
};

TEST_F(DaftGradDescentSingleTest, GradIsSummedMatrix) {
    vector<vector<float>> g;
    f->getGrad("W", &g);
    ASSERT_EQ(g.size(), 1u) << "weight grad must be a single (non-batched) matrix";
    float expected[4] = {1, 2, 1, 2};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(g[0][i], expected[i], 1e-5f) << "grad element " << i;
}

TEST_F(DaftGradDescentSingleTest, SingleSampleUpdate) {
    // beta = -lr (batchSize == 1). lr = 0.5 -> beta = -0.5
    f->gradDescent("W", 0.5f);
    vector<float> w = readWeight(f, "W", 4);
    float expected[4] = {0.5f, 1.0f, 2.5f, 3.0f};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(w[i], expected[i], 1e-5f) << "W element " << i;
}

TEST_F(DaftGradDescentSingleTest, UpdateMatchesGetGradTimesBeta) {
    // White-box: the update must equal W + beta * (the grad we just read).
    vector<vector<float>> g;
    f->getGrad("W", &g);
    float lr = 0.3f;
    float beta = -lr * sqrtf(1) / 1;
    float w0[4] = {1, 2, 3, 4};
    f->gradDescent("W", lr);
    vector<float> w = readWeight(f, "W", 4);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(w[i], w0[i] + beta * g[0][i], 1e-5f) << "W element " << i;
}

TEST_F(DaftGradDescentSingleTest, NonWeightIsNoOp) {
    // gradDescent on a non-WeightsMatrix op must leave weights untouched.
    f->gradDescent("x", 0.5f);
    vector<float> w = readWeight(f, "W", 4);
    float w0[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(w[i], w0[i], 1e-6f) << "W should be unchanged, element " << i;
}

// ---------------------------------------------------------------------------
// gradDescent, two weights: only the named one changes
// ---------------------------------------------------------------------------
class DaftGradDescentTwoWeightsTest : public testing::Test {
protected:
    cublasHandle_t h;
    Function* f;
    void SetUp() override {
        cublasCreate(&h);
        f = new Function(&h);
        f->addOp(Operation::column("x", 2));
        f->addOp(Operation::weightsMatrix("W1", 2, 2));
        f->addOp(Operation::weightsMatrix("W2", 2, 2));
        f->addOp(Operation::matrixProduct("p", "W1", "x", 2, 2, 1));
        f->compile(1);
        f->setValue("W1", {{1, 2, 3, 4}});
        f->setValue("W2", {{5, 6, 7, 8}});
        f->setValue("x", {{1, 1}});
        f->compute();
        f->computeGrad("p");
    }
    void TearDown() override { cublasDestroy(h); delete f; }
};

TEST_F(DaftGradDescentTwoWeightsTest, OnlyTargetWeightChanges) {
    f->gradDescent("W1", 0.5f);
    vector<float> w1 = readWeight(f, "W1", 4);
    vector<float> w2 = readWeight(f, "W2", 4);
    // dL/dW1 = [[1,1],[1,1]], beta = -0.5 -> W1 = {1,2,3,4} - 0.5*{1,1,1,1}
    float e1[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    for (int i = 0; i < 4; i++) EXPECT_NEAR(w1[i], e1[i], 1e-5f) << "W1 element " << i;
    // W2 must be untouched
    float e2[4] = {5, 6, 7, 8};
    for (int i = 0; i < 4; i++) EXPECT_NEAR(w2[i], e2[i], 1e-6f) << "W2 element " << i;
}

// ---------------------------------------------------------------------------
// gradDescent, batchSize > 1 : the summed grad must be applied exactly once
// ---------------------------------------------------------------------------
class DaftGradDescentBatchedTest : public testing::Test {
protected:
    cublasHandle_t h;
    Function* f;
    void SetUp() override {
        cublasCreate(&h);
        f = makeProdNet(&h, 3);
        f->setValue("W", {{1, 2, 3, 4}});
        f->setValue("x", {{1, 0}, {0, 1}, {1, 1}});
        f->compute();
        f->computeGrad("prod");
    }
    void TearDown() override { cublasDestroy(h); delete f; }
};

TEST_F(DaftGradDescentBatchedTest, BatchedGradIsSummed) {
    vector<vector<float>> g;
    f->getGrad("W", &g);
    ASSERT_EQ(g.size(), 1u);
    float expected[4] = {2, 2, 2, 2};   // S0 = 1+0+1, S1 = 0+1+1
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(g[0][i], expected[i], 1e-5f) << "grad element " << i;
}

TEST_F(DaftGradDescentBatchedTest, BatchedUpdateAppliedOnce) {
    float lr = 0.5f;
    float beta = -lr * sqrtf(3) / 3;
    float w0[4] = {1, 2, 3, 4};
    float grad[4] = {2, 2, 2, 2};

    f->gradDescent("W", lr);
    vector<float> w = readWeight(f, "W", 4);

    // Correct: applied exactly once.
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(w[i], w0[i] + beta * grad[i], 1e-4f) << "W element " << i;

    // Regression guard: the old code applied the grad batchSize (3) times.
    float overApplied0 = w0[0] + 3.0f * beta * grad[0];
    EXPECT_GT(std::fabs(w[0] - overApplied0), 1e-2f)
        << "gradDescent appears to have applied the batched grad more than once";
}

// ---------------------------------------------------------------------------
// gradDescent driven from a batchFunction inside batchCompute
// ---------------------------------------------------------------------------
static int g_fnCalls = 0;
static int g_lastIdx = -1;
static int g_lastLen = -1;

static void trainFn(Function* f, int /*idx*/, int /*length*/) {
    f->computeGrad("prod");
    f->gradDescent("W", 0.5f);
}
static void countFn(Function* /*f*/, int idx, int length) {
    g_fnCalls++;
    g_lastIdx = idx;
    g_lastLen = length;
}

class DaftGradDescentInBatchFunctionTest : public testing::Test {
protected:
    cublasHandle_t h;
    Function* f;
    map<string, vector<vector<float>>> inputs;
    void SetUp() override {
        cublasCreate(&h);
        f = makeProdNet(&h, 2);              // batchSize == 2
        f->setValue("W", {{1, 2, 3, 4}});
        // 4 samples -> 2 chunks of 2
        inputs = {{"x", {{1, 0}, {0, 1}, {1, 1}, {2, 0}}}};
    }
    void TearDown() override { cublasDestroy(h); delete f; }
};

TEST_F(DaftGradDescentInBatchFunctionTest, WeightUpdatesAcrossChunks) {
    f->batchCompute({}, {}, inputs, trainFn);
    vector<float> w = readWeight(f, "W", 4);

    float beta = -0.5f * sqrtf(2) / 2;
    // chunk0 S = (1,1); chunk1 S = (3,1)  -> accumulated S = (4,2)
    float totalS[4] = {4, 2, 4, 2};
    float w0[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(w[i], w0[i] + beta * totalS[i], 1e-4f) << "W element " << i;
}

TEST_F(DaftGradDescentInBatchFunctionTest, BatchFunctionInvocationCount) {
    g_fnCalls = 0; g_lastIdx = -1; g_lastLen = -1;
    f->batchCompute({}, {}, inputs, countFn);
    EXPECT_EQ(g_fnCalls, 2) << "expected one call per batchSize chunk (4/2)";
    EXPECT_EQ(g_lastIdx, 2)  << "last chunk should start at sample 2";
    EXPECT_EQ(g_lastLen, 4)  << "length should be the total number of samples";
}

TEST_F(DaftGradDescentInBatchFunctionTest, WeightPersistsAcrossCalls) {
    // Train, then verify a later forward-only pass uses the updated weight.
    f->batchCompute({}, {}, inputs, trainFn);
    vector<float> w1 = readWeight(f, "W", 4);

    map<string, vector<vector<float>>*> results;
    results["prod"] = new vector<vector<float>>;
    map<string, vector<vector<float>>> probe = {{"x", {{1, 0}, {0, 1}}}};
    f->batchCompute(results, {"prod"}, probe);   // 3-arg overload -> no-op batchFunction
    auto& r = *results["prod"];
    ASSERT_EQ(r.size(), 2u);
    // prod = W1 * x : for [1,0] -> [W1[0], W1[2]], for [0,1] -> [W1[1], W1[3]]
    EXPECT_NEAR(r[0][0], w1[0], 1e-4f);
    EXPECT_NEAR(r[0][1], w1[2], 1e-4f);
    EXPECT_NEAR(r[1][0], w1[1], 1e-4f);
    EXPECT_NEAR(r[1][1], w1[3], 1e-4f);
    delete results["prod"];
}

TEST_F(DaftGradDescentInBatchFunctionTest, ForwardOnlyDoesNotChangeWeight) {
    map<string, vector<vector<float>>*> results;
    results["prod"] = new vector<vector<float>>;
    f->batchCompute(results, {"prod"}, inputs);   // no training callback
    vector<float> w = readWeight(f, "W", 4);
    float w0[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(w[i], w0[i], 1e-6f) << "W should be unchanged, element " << i;
    delete results["prod"];
}
