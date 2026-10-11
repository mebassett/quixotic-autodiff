#include <cublas_v2.h>
#include <vector>
#include <random>

#include "../../daft_autodiff/daft_autodiff.h"
#include "../../mnistdata/mnistdata.h"


using namespace std;
using namespace DA;
using namespace MNIST;

const unsigned int NUM_HIDDEN_NODES = 100;

void initializeWeights(vector<float>* m1, vector<float>* m2, float lowerBound, float upperBound) {
    random_device rd;
    mt19937 get(rd());
    uniform_real_distribution<> dis(lowerBound, upperBound);
    auto getRand = [&]() { return dis(rd); };

    for(int i=0; i< m1->size(); i++) 
        (*m1)[i] = getRand();
    for(int i=0; i< m2->size(); i++) 
        (*m2)[i] = getRand();
}

int fromModelOutput(float* out)
{
    float max = *max_element(out, out + OUTPUT_SIZE);
    for (int i { 0 }; i < OUTPUT_SIZE; i++) {
        if (*(out + i) == max)
            return i;
    }
    return -1;
}

int main() {
    float learningRate = 0.025;
    cublasHandle_t cublasH;
    uint BATCHSIZE = 10;

    cublasCreate(&cublasH);
    Function *f;
    f = new Function(&cublasH);
    f->addOp(Operation::column("input", INPUT_SIZE + 1));
    f->addOp(Operation::column("targetInput", OUTPUT_SIZE));
    f->addOp(Operation::weightsMatrix("weights1", NUM_HIDDEN_NODES, INPUT_SIZE + 1));
    f->addOp(Operation::weightsMatrix("weights2", OUTPUT_SIZE, NUM_HIDDEN_NODES));
    f->addOp(Operation::matrixProduct("layer1_output", "weights1", "input", NUM_HIDDEN_NODES, INPUT_SIZE + 1, 1));
    f->addOp(Operation::applyLeakyReLU("layer1_relu", "layer1_output", NUM_HIDDEN_NODES, 1));
    f->addOp(Operation::matrixProduct("prediction", "weights2", "layer1_relu", OUTPUT_SIZE, NUM_HIDDEN_NODES, 1));

    f->addOp(Operation::scalarMultiply("scale","prediction", OUTPUT_SIZE, 1, -1.0f));
    f->addOp(Operation::add("add", "targetInput","scale", OUTPUT_SIZE, 1));
    f->addOp(Operation::innerProduct("ip", "add", "add", OUTPUT_SIZE));
    f->addOp(Operation::scalarMultiply("error", "ip", 1,1, 0.5f));

    f->compile(BATCHSIZE);

    vector<float> weights1 (NUM_HIDDEN_NODES * (INPUT_SIZE + 1));
    vector<float> weights2 (OUTPUT_SIZE * NUM_HIDDEN_NODES);
    
    cout << "init weights...\n";

    float limit = sqrt(6.0f / (INPUT_SIZE + NUM_HIDDEN_NODES));
    initializeWeights(&weights1, &weights2, -limit, limit);

    f->setValue("weights1", {weights1});
    f->setValue("weights2", {weights2});


    Training_Data rows = load_data_from_file("../data/mnist_train.txt", 60000);
    Training_Data testRows = load_data_from_file("../data/mnist_test.txt", 10000);
    int count = 1;
    int numRight = 0;
    int trainingExamples = 0;
    float errorRate;
    numRight = 0;
    errorRate = 0.0;
    trainingExamples = 0;

    for(int i = 0; i<testRows.size(); i += BATCHSIZE) {
        int actualBatchSize = min(BATCHSIZE, (int)testRows.size() - 1);
        vector<Training_Datum> slice ( testRows.begin() + i 
                                     , testRows.begin() + i + actualBatchSize);
        vector<vector<float>> inputs (actualBatchSize);
        vector<vector<float>> targets (actualBatchSize);
        transform(slice.begin(), slice.end(), inputs.begin(), 
            [](auto row) { return vector<float>(begin(row.x), end(row.x)); });
        transform(slice.begin(), slice.end(), targets.begin(), 
            [](auto row) { return vector<float>(begin(row.t), end(row.t)); });

        f->setValue("input", inputs);
        f->setValue("targetInput", targets);

        vector<vector<float>> prediction;
        vector<vector<float>> error;

        f->compute();
        f->getValue("prediction", &prediction);
        f->getValue("error", &error);
        for(int j =0;j<BATCHSIZE;j++){
            int out = fromModelOutput(&(prediction[j][0]));
            errorRate += error[j][0];
            if(out == slice[j].y) numRight++;
        }
    }

    cout << "num right: " << numRight << " / " << testRows.size() << " .\n";
    cout << "model error on test set:" << errorRate << " .\n";

    while (count <= 1) {
        cout << "starting epoch " << count << endl;
        random_device rd;
        mt19937 g(rd());
        shuffle(rows.begin(), rows.end(), g);
        for(int i = 0; i<rows.size(); i += BATCHSIZE) {
            f->resetGrad();

            int actualBatchSize = min(BATCHSIZE, (int)rows.size() - 1);
            vector<Training_Datum> slice ( rows.begin() + i 
                                         , rows.begin() + i + actualBatchSize);
            vector<vector<float>> inputs (actualBatchSize);
            vector<vector<float>> targets (actualBatchSize);
            transform(slice.begin(), slice.end(), inputs.begin(), 
                [](auto row) { return vector<float>(begin(row.x), end(row.x)); });
            transform(slice.begin(), slice.end(), targets.begin(), 
                [](auto row) { return vector<float>(begin(row.t), end(row.t)); });

            f->setValue("input", inputs);
            f->setValue("targetInput", targets);

            f->compute();
            f->computeGrad("error");

            f->gradDescent("weights1", learningRate);
            f->gradDescent("weights2", learningRate);
            trainingExamples += actualBatchSize;
            if (trainingExamples % 10000 == 0)
                cout << "done " << trainingExamples << " so far." << endl;
        }
        numRight = 0;
        errorRate = 0.0;
        trainingExamples = 0;
        for(int i = 0; i<testRows.size(); i += BATCHSIZE) {
            int actualBatchSize = min(BATCHSIZE, (int)testRows.size() - 1);
            vector<Training_Datum> slice ( testRows.begin() + i 
                                         , testRows.begin() + i + actualBatchSize);
            vector<vector<float>> inputs (actualBatchSize);
            vector<vector<float>> targets (actualBatchSize);
            transform(slice.begin(), slice.end(), inputs.begin(), 
                [](auto row) { return vector<float>(begin(row.x), end(row.x)); });
            transform(slice.begin(), slice.end(), targets.begin(), 
                [](auto row) { return vector<float>(begin(row.t), end(row.t)); });

            f->setValue("input", inputs);
            f->setValue("targetInput", targets);

            vector<vector<float>> prediction;
            vector<vector<float>> error;

            f->compute();
            f->getValue("prediction", &prediction);
            f->getValue("error", &error);
            for(int j =0;j<BATCHSIZE;j++){
                int out = fromModelOutput(&(prediction[j][0]));
                errorRate += error[j][0];
                if(out == slice[j].y) numRight++;
            }
        }
        cout << "num right: " << numRight << " / " << testRows.size() << " .\n";
        cout << "model error on test set:" << errorRate << " .\n";
        count++;
    }



    cublasDestroy(cublasH);
    delete f;
    



    
}
