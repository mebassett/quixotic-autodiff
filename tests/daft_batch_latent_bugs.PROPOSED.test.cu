// PROPOSED tests for latent batch bugs found by static analysis.
// These are EXPECTED TO FAIL until the underlying ops are fixed.
#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace DA;

// ---------------------------------------------------------------------------
// PROPOSAL 1 — Concat with UNEVEN target sizes.
// compute(Concat) and computeGrad(Concat) launch with
//     gd.x = ceil(targetSizes[0] / 16)
// so any target BIGGER than targets[0] is silently dropped (forward) or
// left with zero grad (backward).
// Here targets[0] is size 4, targets[1] is size 16.
// ---------------------------------------------------------------------------
class ConcatUnevenTest : public testing::Test {
protected:
    cublasHandle_t h;
    Function* f;
    void SetUp() override {
        cublasCreate(&h);
        f = new Function(&h);
        f->addOp(Operation::inputMatrix("small", 2, 2));    // resultSize 4
        f->addOp(Operation::inputMatrix("big", 4, 6));      // resultSize 24
        f->addOp(Operation::concat("cat", {"small", "big"}, 28));
        f->addOp(Operation::innerProduct("ip", "cat", "cat", 28));
        f->compile(2);
        f->setValue("small", { {1,2,3,4}, {5,6,7,8} });
        f->setValue("big", {
            {1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5, 6,6,6,6},
            {5,5,5,5, 6,6,6,6, 7,7,7,7, 8,8,8,8, 9,9,9,9, 10,10,10,10}
        });
        f->compute();
        f->computeGrad("ip");
    }
    void TearDown() override { cublasDestroy(h); delete f; }
};

TEST_F(ConcatUnevenTest, ForwardContainsAllOfBothTargets) {
    vector<vector<float>> cat;
    f->getValue("cat", &cat);
    // batch 0 expected: [1,2,3,4 | 1x4, 2x4, 3x4, 4x4, 5x4, 6x4]
    float e0[28] = {1,2,3,4, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5, 6,6,6,6};
    float e1[28] = {5,6,7,8, 5,5,5,5, 6,6,6,6, 7,7,7,7, 8,8,8,8, 9,9,9,9, 10,10,10,10};
    for (int i = 0; i < 28; i++) {
        EXPECT_NEAR(cat[0][i], e0[i], 1e-6f) << "batch0 elem " << i;
        EXPECT_NEAR(cat[1][i], e1[i], 1e-6f) << "batch1 elem " << i;
    }
}

TEST_F(ConcatUnevenTest, BackwardReachesAllOfBothTargets) {
    vector<vector<float>> bg;
    f->getGrad("big", &bg);
    // d||cat||^2 / d(big) = 2*big, for both batches
    for (int b = 0; b < 2; b++) {
        float base = (b == 0) ? 1.f : 5.f;
        for (int i = 0; i < 24; i++) {
            float expect = 2.f * (base + i / 4);
            EXPECT_NEAR(bg[b][i], expect, 1e-5f)
                << "big grad batch " << b << " elem " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// PROPOSAL 2 — Overlapping max-pool gradient must SUM.
// 3x3 input, 2x2 window, stride 1 -> 2x2 output; windows OVERLAP.
// Centre cell (1,1) = 9 is the max of ALL four windows, so
//     dL/d(in[1][1]) = seed(w00)+seed(w01)+seed(w10)+seed(w11).
// doMaxPoolGrad ASSIGNS (result[mIndex] = ...) instead of accumulating,
// so the result is one window's seed (nondeterministic race), not the sum.
// bs=2 also catches batch-stride issues in the same kernel.
// ---------------------------------------------------------------------------
class OverlappingPoolGradTest : public testing::Test {
protected:
    cublasHandle_t h;
    Function* f;
    void SetUp() override {
        cublasCreate(&h);
        f = new Function(&h);
        f->addOp(Operation::inputMatrix("in", 3, 3));
        f->addOp(Operation::maxPool("pool", "in", 2, 2, 1, 1, 3, 3)); // 2x2 out
        f->addOp(Operation::scalarMultiply("scaled", "pool", 2, 2, 3.0f));
        f->compile(2);
        // batch0 centre 9, batch1 centre 7 (max of all windows both)
        f->setValue("in", {
            {1, 1, 1,  1, 9, 1,  1, 1, 1},
            {1, 1, 1,  1, 7, 1,  1, 1, 1}
        });
        f->compute();
        // seed = [1,2,3,4] per batch (via scalar of 3 -> 3,6,9,12 downstream?
        // easier: seed "pool" directly)
        float seed[8] = {1,2,3,4,  10,20,30,40};
        float* d_seed;
        cudaMalloc(&d_seed, 8 * sizeof(float));
        cudaMemcpy(d_seed, seed, 8 * sizeof(float), cudaMemcpyHostToDevice);
        f->computeGrad("pool", d_seed);
        cudaFree(d_seed);
    }
    void TearDown() override { cublasDestroy(h); delete f; }
};

TEST_F(OverlappingPoolGradTest, CentreGetsSumOfAllWindows) {
    vector<vector<float>> g;
    f->getGrad("in", &g);
    // centre (1,1) = index 4 must equal sum of all four window seeds
    EXPECT_NEAR(g[0][4], 1+2+3+4, 1e-5f) << "batch0 centre grad";
    EXPECT_NEAR(g[1][4], 10+20+30+40, 1e-5f) << "batch1 centre grad";
    // the four corner cells each belong to exactly one window:
    EXPECT_NEAR(g[0][0], 1, 1e-5f);   // window (0,0)
    EXPECT_NEAR(g[0][2], 2, 1e-5f);   // window (0,1)
    EXPECT_NEAR(g[0][6], 3, 1e-5f);   // window (1,0)
    EXPECT_NEAR(g[0][8], 4, 1e-5f);   // window (1,1)
}

// ---------------------------------------------------------------------------
// PROPOSAL 3 — Single-target Concat allocation.
// compile() SKIPS (continue) size accounting for a concat with ONE target,
// but the address-assignment loop still gives it memory -> d_value is
// under-allocated by (result+grad) * batchSize and the LAST op in the list
// reads/writes past the end of the cudaMalloc'd block.
// (compute-sanitizer memcheck should flag it; even without sanitizer this
// test can catch corruption by placing a sensitive op AFTER the concat.)
// ---------------------------------------------------------------------------
class SingleTargetConcatTest : public testing::Test {
protected:
    cublasHandle_t h;
    Function* f;
    void SetUp() override {
        cublasCreate(&h);
        f = new Function(&h);
        f->addOp(Operation::inputMatrix("a", 4, 4));            // resultSize 16
        f->addOp(Operation::weightsMatrix("W", 4, 4));
        f->addOp(Operation::concat("cat", {"a"}, 16));           // single target
        f->addOp(Operation::matrixProduct("out", "W", "cat", 4, 4, 1));
        f->compile(3);
        f->setValue("a", { {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16},
                           {1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4},
                           {2,0,0,0, 0,2,0,0, 0,0,2,0, 0,0,0,2} });
        f->setValue("W", { {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1} }); // identity
        f->compute();
    }
    void TearDown() override { cublasDestroy(h); delete f; }
};

TEST_F(SingleTargetConcatTest, ResultsSurviveAcrossAllBatches) {
    vector<vector<float>> out;
    f->getValue("out", &out);
    float e0[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float e1[16] = {1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4};
    float e2[16] = {2,0,0,0, 0,2,0,0, 0,0,2,0, 0,0,0,2};
    for (int i = 0; i < 16; i++) {
        EXPECT_NEAR(out[0][i], e0[i], 1e-5f) << "batch0 elem " << i;
        EXPECT_NEAR(out[1][i], e1[i], 1e-5f) << "batch1 elem " << i;
        EXPECT_NEAR(out[2][i], e2[i], 1e-5f) << "batch2 elem " << i;
    }
}
