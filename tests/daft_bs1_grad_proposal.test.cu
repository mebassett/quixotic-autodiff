#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace DA;

// ===========================================================================
// PROPOSAL A.1 — Convolution INPUT gradient with PADDING at bs>1.
//
// This is the regression test for the doCopyWithoutPadding batch-stride bug:
//   int srcSize = (rows+rowPadding)*(cols+colPadding);
// should be (rows+2*rowPadding)*(cols+2*colPadding) == paddedInputSize.
// With padding==0 the two are equal, which is why the existing
// DaftBatchConvolutionGradTest (no padding) passes.
//
// 2x2 input, 2x2 all-ones kernel, pad 1, stride 1  ->  3x3 output, bs=2.
// seed = 1s (from computeGrad("conv")).
// Correct input grad = [[4,4],[4,4]] for BOTH batches.
// With the bug, batch 1 reads the wrong padded offset -> [[2,1],[2,1]].
// ===========================================================================
class PropConvPadGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::inputMatrix("input", 2, 2));
        f->addOp(Operation::weightsMatrix("kernel", 2, 2));
        f->addOp(Operation::convolution("conv", "input", "kernel",
                                       1, 1, 1, 1,   // rowPad, rowSkip, colPad, colSkip
                                       2, 2, 2, 2));
        f->compile(2);
        f->setValue("kernel", {{1, 1, 1, 1}});
        f->setValue("input", {
            {1, 2, 3, 4},
            {5, 6, 7, 8}
        });
        f->compute();
        f->computeGrad("conv");
    }
    void TearDown() override { cublasDestroy(cublasH); delete f; }
};

TEST_F(PropConvPadGradTest, InputGradBothBatches) {
    vector<vector<float>> ig;
    f->getGrad("input", &ig);
    // batch 0 (correct even with bug)
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(ig[0][i], 4) << "conv pad input-grad batch0 elem " << i;
    // batch 1 (CATCHES the doCopyWithoutPadding bug)
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(ig[1][i], 4) << "conv pad input-grad batch1 elem " << i;
}

TEST_F(PropConvPadGradTest, KernelGradSummed) {
    vector<vector<float>> kg;
    f->getGrad("kernel", &kg);
    // per-batch kernel grad: b0=[10,10,10,10], b1=[26,26,26,26] -> sum [36,...]
    for (int i = 0; i < 4; i++)
        EXPECT_EQ(kg[0][i], 36) << "conv pad kernel-grad elem " << i;
}

// ===========================================================================
// PROPOSAL B.4 — MaxPool gradient at bs>1 (the existing "batch" test is
// mislabeled: it compiles with batchSize=1). 2x2 input, 2x2 pool stride 1
// -> 1x1, scalarMultiply(seed=5), bs=3, max at a different cell per batch.
// ===========================================================================
class PropMaxPoolGradTest : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::inputMatrix("input", 2, 2));
        f->addOp(Operation::maxPool("mp", "input", 2, 2, 1, 1, 2, 2));
        f->addOp(Operation::scalarMultiply("smp", "mp", 1, 1, 5));
        f->compile(3);
        f->setValue("input", {
            {1, 1, 1, 4},   // max at (1,1)
            {4, 1, 1, 1},   // max at (0,0)
            {1, 4, 1, 1}    // max at (0,1)
        });
        f->compute();
        f->computeGrad("smp");
    }
    void TearDown() override { cublasDestroy(cublasH); delete f; }
};

TEST_F(PropMaxPoolGradTest, GradPerBatch) {
    vector<vector<float>> g;
    f->getGrad("input", &g);
    float exp[3][4] = {
        {0, 0, 0, 5},
        {5, 0, 0, 0},
        {0, 5, 0, 0}
    };
    for (int b = 0; b < 3; b++)
        for (int i = 0; i < 4; i++)
            EXPECT_EQ(g[b][i], exp[b][i]) << "maxpool grad batch " << b << " elem " << i;
}

// ===========================================================================
// PROPOSAL C.8 — Concat gradient at bs=3 (the existing test is capped at
// bs=2 with a NOTE about a suspected scratch overlap). 2x4 + 2x4 -> 16,
// innerProduct(c,c) -> ||c||^2 per batch, so each target grad = 2*its values.
// ===========================================================================
class PropConcatGradBS3Test : public testing::Test {
protected:
    cublasHandle_t cublasH;
    Function* f;
    void SetUp() override {
        cublasCreate(&cublasH);
        f = new Function(&cublasH);
        f->addOp(Operation::inputMatrix("a", 2, 4));
        f->addOp(Operation::inputMatrix("b", 2, 4));
        f->addOp(Operation::concat("c", {"a", "b"}, 16));
        f->addOp(Operation::innerProduct("ip", "c", "c", 16));
        f->compile(3);
        f->setValue("a", {
            {1, 2, 3, 4, 5, 6, 7, 8},
            {0, 0, 0, 0, 1, 1, 1, 1},
            {9, 9, 0, 0, 0, 0, 9, 9}
        });
        f->setValue("b", {
            {10, 10, 10, 10, 10, 10, 10, 10},
            {2, 2, 2, 2, 2, 2, 2, 2},
            {1, 0, 0, 0, 0, 0, 0, 1}
        });
        f->compute();
        f->computeGrad("ip");
    }
    void TearDown() override { cublasDestroy(cublasH); delete f; }
};

TEST_F(PropConcatGradBS3Test, GradPerBatch) {
    vector<vector<float>> ag, bg;
    f->getGrad("a", &ag);
    f->getGrad("b", &bg);
    float av[3][8] = {
        {1, 2, 3, 4, 5, 6, 7, 8},
        {0, 0, 0, 0, 1, 1, 1, 1},
        {9, 9, 0, 0, 0, 0, 9, 9}
    };
    float bv[3][8] = {
        {10, 10, 10, 10, 10, 10, 10, 10},
        {2, 2, 2, 2, 2, 2, 2, 2},
        {1, 0, 0, 0, 0, 0, 0, 1}
    };
    for (int b = 0; b < 3; b++) {
        for (int i = 0; i < 8; i++) {
            EXPECT_EQ(ag[b][i], 2 * av[b][i]) << "concat a grad b" << b << " e" << i;
            EXPECT_EQ(bg[b][i], 2 * bv[b][i]) << "concat b grad b" << b << " e" << i;
        }
    }
}
