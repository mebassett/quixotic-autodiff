#include "../daft_autodiff/daft_autodiff.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstdio>

using namespace DA;

// ---------------------------------------------------------------------------
// Diagnostic: is the bs=2 gradient ADDITIVE?
//   grad_{bs=2}([A,B])  should equal  grad_{bs=1}(A) + grad_{bs=1}(B)
// If not, some op is not composing its per-sample gradient linearly.
// ---------------------------------------------------------------------------

static const float A16[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const float B16[16] = {4, 0, 1, 0, 3, 2, 0, 5, 1, 1, 2, 0, 4, 3, 0, 2};

// mini LeNet: two 4x4 "channels" -> conv(2x2) -> maxpool(2x2,s1) -> concat(8)
//             -> fc(1x8) -> loss = fc^2
static Function* buildMini(cublasHandle_t* h, int bs) {
    Function* f = new Function(h);
    f->addOp(Operation::inputMatrix("in1", 4, 4));
    f->addOp(Operation::inputMatrix("in2", 4, 4));
    f->addOp(Operation::weightsMatrix("k1", 2, 2));
    f->addOp(Operation::weightsMatrix("k2", 2, 2));
    f->addOp(Operation::convolution("c1", "in1", "k1", 0,1,0,1, 4,4, 2,2)); // 3x3
    f->addOp(Operation::convolution("c2", "in2", "k2", 0,1,0,1, 4,4, 2,2)); // 3x3
    f->addOp(Operation::maxPool("p1", "c1", 2,2, 1,1, 3,3));                 // 2x2
    f->addOp(Operation::maxPool("p2", "c2", 2,2, 1,1, 3,3));                 // 2x2
    f->addOp(Operation::concat("flat", {"p1","p2"}, 8));
    f->addOp(Operation::weightsMatrix("W", 1, 8));
    f->addOp(Operation::matrixProduct("fc", "W", "flat", 1, 8, 1));          // 1x1
    f->addOp(Operation::innerProduct("loss", "fc", "fc", 1));
    f->compile(bs);
    return f;
}

// read a WeightsMatrix's single (summed) grad as a flat vector of n
static std::vector<float> readWGrad(Function* f, const std::string& name, int n) {
    std::vector<std::vector<float>> g;
    f->getGrad(name, &g);
    std::vector<float> out(n);
    for (int i = 0; i < n; i++) out[i] = g[0][i];
    return out;
}

static void setOne(Function* f, const char* nm, const float* v, int n) {
    std::vector<std::vector<float>> one(1, std::vector<float>(v, v + n));
    f->setValue(nm, one);
}

TEST(MiniAdditivity, ConvKernelAndFc) {
    // ---- grad for A alone (bs=1) ----
    cublasHandle_t h; cublasCreate(&h);
    Function* fA = buildMini(&h, 1);
    setOne(fA, "in1", A16, 16); setOne(fA, "in2", A16, 16);
    setOne(fA, "k1", (const float[]){1, -0.5f, 0.25f, 2}, 4);
    setOne(fA, "k2", (const float[]){-1, 0.5f, 1.5f, 0.1f}, 4);
    setOne(fA, "W", (const float[]){0.3f,-0.2f,0.7f,0.1f,-0.4f,0.9f,0.2f,-0.1f}, 8);
    fA->compute(); fA->computeGrad("loss");
    std::vector<float> gk1A = readWGrad(fA, "k1", 4);
    std::vector<float> gk2A = readWGrad(fA, "k2", 4);
    std::vector<float> gWA  = readWGrad(fA, "W", 8);
    delete fA;

    // ---- grad for B alone (bs=1) ----
    Function* fB = buildMini(&h, 1);
    setOne(fB, "in1", B16, 16); setOne(fB, "in2", B16, 16);
    setOne(fB, "k1", (const float[]){1, -0.5f, 0.25f, 2}, 4);
    setOne(fB, "k2", (const float[]){-1, 0.5f, 1.5f, 0.1f}, 4);
    setOne(fB, "W", (const float[]){0.3f,-0.2f,0.7f,0.1f,-0.4f,0.9f,0.2f,-0.1f}, 8);
    fB->compute(); fB->computeGrad("loss");
    std::vector<float> gk1B = readWGrad(fB, "k1", 4);
    std::vector<float> gk2B = readWGrad(fB, "k2", 4);
    std::vector<float> gWB  = readWGrad(fB, "W", 8);
    delete fB;

    // ---- grad for [A,B] together (bs=2) ----
    Function* fAB = buildMini(&h, 2);
    fAB->setValue("in1", {{A16, A16+16}, {B16, B16+16}});
    fAB->setValue("in2", {{A16, A16+16}, {B16, B16+16}});
    setOne(fAB, "k1", (const float[]){1, -0.5f, 0.25f, 2}, 4);
    setOne(fAB, "k2", (const float[]){-1, 0.5f, 1.5f, 0.1f}, 4);
    setOne(fAB, "W", (const float[]){0.3f,-0.2f,0.7f,0.1f,-0.4f,0.9f,0.2f,-0.1f}, 8);
    fAB->compute(); fAB->computeGrad("loss");
    std::vector<float> gk1AB = readWGrad(fAB, "k1", 4);
    std::vector<float> gk2AB = readWGrad(fAB, "k2", 4);
    std::vector<float> gWAB  = readWGrad(fAB, "W", 8);
    delete fAB;
    cublasDestroy(h);

    auto check = [](const char* nm, const std::vector<float>& sum,
                    const std::vector<float>& a, const std::vector<float>& b) {
        for (size_t i = 0; i < sum.size(); i++) {
            float expect = a[i] + b[i];
            EXPECT_NEAR(sum[i], expect, 1e-2f * std::max(1.0f, std::abs(expect)))
                << nm << " elem " << i << ": batched=" << sum[i]
                << " sum=" << expect << " (A=" << a[i] << " B=" << b[i] << ")";
        }
    };
    check("k1", gk1AB, gk1A, gk1B);
    check("k2", gk2AB, gk2A, gk2B);
    check("W",  gWAB,  gWA,  gWB);
}
