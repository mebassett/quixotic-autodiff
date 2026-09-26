#include <cublas_v2.h>
#include <vector>
#include <random>
#include <chrono>

#include "../../daft_autodiff/daft_autodiff.h"
#include "../../mnistdata/mnistdata.h"


using namespace std;
using namespace DA;
using namespace MNIST;

void initialize_weights(Function* f, const string& name, float lower_bound, float upper_bound) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(lower_bound, upper_bound);
    auto get_rand = [&]() { return dis(rd); };

    // Find the operation to get its size
    for (auto& op : f->ops) {
        if (op.name == name) {
            vector<float> weights(op.rows * op.cols);
            for (int i = 0; i < weights.size(); i++) {
                weights[i] = get_rand();
            }
            f->setValue(name, {weights});
            break;
        }
    }
}

int fromModelOutput(const float* out) {
    float max = *max_element(out, out + OUTPUT_SIZE);
    for (int i { 0 }; i < OUTPUT_SIZE; i++) {
        if (*(out + i) == max)
            return i;
    }
    return -1;
}

float learningRate = 0.025;
void batchTrainFunction(Function* f, int idx, int length) {
  f->computeGrad("loss");
  
  // Update weights
  for (int i = 0; i < 6; i++) {
      f->gradDescent("c1-kernel-" + to_string(i), learningRate);
  }
  for (int i = 0; i < 16; i++) {
      f->gradDescent("c3-kernel-" + to_string(i), learningRate);
  }
  f->gradDescent("fc1-weights", learningRate);
  f->gradDescent("fc2-weights", learningRate);
  f->gradDescent("output-weights", learningRate);
  if(idx % 10000 == 0) {
    cout << "Completed " << idx << " examples out of " << length << "." << endl;
  }
}

int main() {
    // see https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf for architecture.
    cout << "Building LeNet...\n";

    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    Function f = Function(&cublasH);

    f.addOp(Operation::inputMatrix("featureInput", 28, 28));
    f.addOp(Operation::column("targetInput", 10));

    // C1 layer - 6 feature maps with 5x5 kernels
    for (int i = 0; i < 6; i++) {
        f.addOp(Operation::weightsMatrix("c1-kernel-" + to_string(i), 5, 5));
    }
    for (int i = 0; i < 6; i++) {
        f.addOp(Operation::convolution("c1-conv-" + to_string(i)
                , "featureInput"
                , "c1-kernel-" + to_string(i)
                , 2, 1, 2, 1
                , 28, 28
                , 5, 5));
    }
    for (int i = 0; i < 6; i++) {
        f.addOp(Operation::applyLeakyReLU("c1-relu-" + to_string(i)
                , "c1-conv-" + to_string(i)
                , 28, 28));
    }
    // S2 layer - 2x2 max pooling
    for (int i = 0; i < 6; i++) {
        f.addOp(Operation::maxPool("s2-pool-" + to_string(i)
                , "c1-relu-" + to_string(i)
                , 2, 2, 2, 2
                , 28, 28));
    }

    // C3 layer - 16 feature maps with 5x5 kernels
    for (int i = 0; i < 16; i++) {
        f.addOp(Operation::weightsMatrix("c3-kernel-" + to_string(i), 5, 5));
    }

    // C3 connections - first 6 are combinations of 3 continuous channels
    int c3TripleIndex[6][3] = { { 0, 1, 2 }, { 1, 2, 3 }, { 2, 3, 4 },
        { 3, 4, 5 }, { 4, 5, 0 }, { 5, 0, 1 } };
    for (int i = 0; i < 6; i++) {
        int r = c3TripleIndex[i][0];
        int s = c3TripleIndex[i][1];
        int t = c3TripleIndex[i][2];
        
        f.addOp(Operation::convolution("c3-conv-" + to_string(i) + "-" + to_string(r)
                , "s2-pool-" + to_string(r)
                , "c3-kernel-" + to_string(i)
                , 0, 1, 0, 1
                , 14, 14
                , 5, 5));
        f.addOp(Operation::convolution("c3-conv-" + to_string(i) + "-" + to_string(s)
                , "s2-pool-" + to_string(s)
                , "c3-kernel-" + to_string(i)
                , 0, 1, 0, 1
                , 14, 14
                , 5, 5));
        f.addOp(Operation::convolution("c3-conv-" + to_string(i) + "-" + to_string(t)
                , "s2-pool-" + to_string(t)
                , "c3-kernel-" + to_string(i)
                , 0, 1, 0, 1
                , 14, 14
                , 5, 5));
        f.addOp(Operation::add("c3-add1-" + to_string(i)
                , "c3-conv-" + to_string(i) + "-" + to_string(r)
                , "c3-conv-" + to_string(i) + "-" + to_string(s)
                , 10, 10));
        f.addOp(Operation::add("c3-layer-" + to_string(i)
                , "c3-add1-" + to_string(i)
                , "c3-conv-" + to_string(i) + "-" + to_string(t)
                , 10, 10));
    }

    // Next 9 C3 connections - combinations of 4 channels
    int c3QuadIndex[9][4] = { { 0, 1, 2, 3 }, { 1, 2, 3, 4 }, { 2, 3, 4, 5 },
        { 3, 4, 5, 0 }, { 4, 5, 0, 1 }, { 5, 0, 1, 2 },
        { 0, 1, 3, 4 }, { 1, 2, 4, 5 }, { 0, 2, 3, 5 } };
    for (int i = 6; i < 15; i++) {
        int r = c3QuadIndex[i - 6][0];
        int s = c3QuadIndex[i - 6][1];
        int t = c3QuadIndex[i - 6][2];
        int u = c3QuadIndex[i - 6][3];
        
        f.addOp(Operation::convolution("c3-conv-" + to_string(i) + "-" + to_string(r)
                , "s2-pool-" + to_string(r)
                , "c3-kernel-" + to_string(i)
                , 0, 1, 0, 1
                , 14, 14
                , 5, 5));
        f.addOp(Operation::convolution("c3-conv-" + to_string(i) + "-" + to_string(s)
                , "s2-pool-" + to_string(s)
                , "c3-kernel-" + to_string(i)
                , 0, 1, 0, 1
                , 14, 14
                , 5, 5));
        f.addOp(Operation::convolution("c3-conv-" + to_string(i) + "-" + to_string(t)
                , "s2-pool-" + to_string(t)
                , "c3-kernel-" + to_string(i)
                , 0, 1, 0, 1
                , 14, 14
                , 5, 5));
        f.addOp(Operation::convolution("c3-conv-" + to_string(i) + "-" + to_string(u)
                , "s2-pool-" + to_string(u)
                , "c3-kernel-" + to_string(i)
                , 0, 1, 0, 1
                , 14, 14
                , 5, 5));
        f.addOp(Operation::add("c3-add1-" + to_string(i)
                , "c3-conv-" + to_string(i) + "-" + to_string(r)
                , "c3-conv-" + to_string(i) + "-" + to_string(s)
                , 10, 10));
        f.addOp(Operation::add("c3-add2-" + to_string(i)
                , "c3-conv-" + to_string(i) + "-" + to_string(t)
                , "c3-conv-" + to_string(i) + "-" + to_string(u)
                , 10, 10));
        f.addOp(Operation::add("c3-layer-" + to_string(i)
                , "c3-add1-" + to_string(i)
                , "c3-add2-" + to_string(i)
                , 10, 10));
    }

    // Last C3 connection - all 6 channels
    for (int j = 0; j < 6; j++) {
        f.addOp(Operation::convolution("c3-conv-15-" + to_string(j)
                , "s2-pool-" + to_string(j)
                , "c3-kernel-15"
                , 0, 1, 0, 1
                , 14, 14
                , 5, 5));
    }
    f.addOp(Operation::add("c3-add1-15"
            , "c3-conv-15-0"
            , "c3-conv-15-1"
            , 10, 10));
    f.addOp(Operation::add("c3-add2-15"
            , "c3-conv-15-2"
            , "c3-conv-15-3"
            , 10, 10));
    f.addOp(Operation::add("c3-add3-15"
            , "c3-conv-15-4"
            , "c3-conv-15-5"
            , 10, 10));
    f.addOp(Operation::add("c3-add4-15"
            , "c3-add1-15"
            , "c3-add2-15"
            , 10, 10));
    f.addOp(Operation::add("c3-layer-15"
            , "c3-add4-15"
            , "c3-add3-15"
            , 10, 10));

    // Apply ReLU to all C3 layers
    for (int i = 0; i < 16; i++) {
        f.addOp(Operation::applyLeakyReLU("c3-relu-" + to_string(i)
                , "c3-layer-" + to_string(i)
                , 10, 10));
    }

    // S4 layer - 2x2 max pooling
    for (int i = 0; i < 16; i++) {
        f.addOp(Operation::maxPool("s4-pool-" + to_string(i)
                , "c3-relu-" + to_string(i)
                , 2, 2, 2, 2
                , 10, 10));
    }

    // Flatten S4 outputs into one vector (16 * 5 * 5 = 400)
    vector<string> s4_targets;
    for (int i = 0; i < 16; i++) {
        s4_targets.push_back("s4-pool-" + to_string(i));
    }
    f.addOp(Operation::concat("s4-flattened", s4_targets, 400));

    // Fully connected layers
    f.addOp(Operation::weightsMatrix("fc1-weights", 120, 400));
    f.addOp(Operation::matrixProduct("fc1-output", "fc1-weights", "s4-flattened", 120, 400, 1));
    f.addOp(Operation::applyLeakyReLU("fc1-relu", "fc1-output", 120, 1));

    f.addOp(Operation::weightsMatrix("fc2-weights", 84, 120));
    f.addOp(Operation::matrixProduct("fc2-output", "fc2-weights", "fc1-relu", 84, 120, 1));
    f.addOp(Operation::applyLeakyReLU("fc2-relu", "fc2-output", 84, 1));

    f.addOp(Operation::weightsMatrix("output-weights", 10, 84));
    f.addOp(Operation::matrixProduct("prediction", "output-weights", "fc2-relu", 10, 84, 1));

    // Loss function
    f.addOp(Operation::scalarMultiply("neg-prediction", "prediction", 10, 1, -1.0f));
    f.addOp(Operation::add("error-term", "targetInput", "neg-prediction", 10, 1));
    f.addOp(Operation::innerProduct("squared-error", "error-term", "error-term", 10));
    f.addOp(Operation::scalarMultiply("loss", "squared-error", 1, 1, 0.5f));

    f.compile();

    cout << "Model built, now initializing weights...\n";
    
    // Initialize all weights
    for (int i = 0; i < 6; i++) {
        initialize_weights(&f, "c1-kernel-" + to_string(i), -0.05, 0.05);
    }
    for (int i = 0; i < 16; i++) {
        initialize_weights(&f, "c3-kernel-" + to_string(i), -0.05, 0.05);
    }
    initialize_weights(&f, "fc1-weights", -0.05, 0.05);
    initialize_weights(&f, "fc2-weights", -0.05, 0.05);
    initialize_weights(&f, "output-weights", -0.05, 0.05);

    cout << "Loading training and test data...\n";
    Training_Data rows = load_data_from_file_no_bias("../data/mnist_train.txt", 60000);
    Training_Data testRows = load_data_from_file_no_bias("../data/mnist_test.txt", 10000);

    cout << "Testing initial model performance...\n";
    int numRight = 0;
    float errorRate = 0.0;

    map<string, vector<vector<float>>> testInputs;
    
    for (auto row : testRows) {
        vector<float> input(begin(row.x), end(row.x));
        vector<float> target(begin(row.t), end(row.t));

        testInputs["featureInput"].push_back(input);
        testInputs["targetInput"].push_back(target);
    }

    map<string, vector<vector<float>>> trainInputs;
    
    for (auto row : rows) {
        vector<float> input(begin(row.x), end(row.x));
        vector<float> target(begin(row.t), end(row.t));

        trainInputs["featureInput"].push_back(input);
        trainInputs["targetInput"].push_back(target);
    }



    map<string, vector<vector<float>>*> testResults;
    testResults["prediction"] = new vector<vector<float>>();
    testResults["loss"] = new vector<vector<float>>();

    f.batchCompute(testResults, {"prediction", "loss"}, testInputs);

    for(int i=0;i<testRows.size();i++){
        vector<float>& prediction = (*(testResults["prediction"]))[i];
        float& loss = (*(testResults["loss"]))[i][0];

        int out = fromModelOutput(&(prediction[0]));
        errorRate += loss;
        if (out == testRows[i].y) numRight++;

    }
    cout << "Initial accuracy: " << numRight << " / " << testRows.size() << "\n";
    cout << "Initial error: " << errorRate << "\n\n";

    cout << "Starting training...\n";
    int epochs = 1;
    auto startTime = chrono::steady_clock::now();
    
    while (epochs <= 20) {
        auto epochStartTime = chrono::steady_clock::now();
        cout << "Starting epoch " << epochs << "...\n";

        f.batchCompute({}, {}, trainInputs, batchTrainFunction);
        
        
        auto epochEndTime = chrono::steady_clock::now();
        auto elapsed = epochEndTime - startTime;
        auto lapped = epochEndTime - epochStartTime;
        
        cout << "Testing after epoch " << epochs << "...\n";
        cout << "Elapsed time: " << chrono::duration_cast<chrono::seconds>(elapsed).count() << " s\n";
        cout << "Epoch time: " << chrono::duration_cast<chrono::seconds>(lapped).count() << " s\n";
        
        numRight = 0;
        errorRate = 0.0;
        testResults["prediction"]->clear();
        testResults["loss"]->clear();

        f.batchCompute(testResults, {"prediction", "loss"}, testInputs);

        for(int i=0;i<testRows.size();i++){
            vector<float>& prediction = (*(testResults["prediction"]))[i];
            float& loss = (*(testResults["loss"]))[i][0];

            int out = fromModelOutput(&(prediction[0]));
            errorRate += loss;
            if (out == testRows[i].y) numRight++;

        }
        
        
        cout << "Accuracy: " << numRight << " / " << testRows.size() << "\n";
        cout << "Error: " << errorRate << "\n\n";
        
        epochs++;
    }

    cublasDestroy(cublasH);
    return 0;
}


