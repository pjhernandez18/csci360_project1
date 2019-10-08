#include "FeedForwardNetwork.h"
#include <iostream> 
#include <random> 
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <cmath> 
#include <math.h>

using namespace std;

void FeedForwardNetwork::init(int seed) {
    srand(seed);
    hiddenLayerWeights.resize(inputLayerSize);
    for (size_t i = 0; i < inputLayerSize; i++) {
        hiddenLayerWeights[i].resize(hiddenLayerSize);
        for (size_t j = 0; j < hiddenLayerSize; j++) {
            // initialized random weights from -0.5 to 0.5
            hiddenLayerWeights[i][j] = (double) ((rand() % 100 + 1) * 1.0 / 100) - 0.5;
            //cout << hiddenLayerWeights[i][j] << " ";
        }
        //cout << endl;
    }
    outputLayerWeights.resize(hiddenLayerSize);
    for (size_t i = 0; i < hiddenLayerSize; i++) {
        outputLayerWeights[i].resize(outputLayerSize);
        for (size_t j = 0; j < outputLayerSize; j++) {
            // initialized random weights from -0.5 to 0.5
            outputLayerWeights[i][j] = (double) ((rand() % 100 + 1) * 1.0 / 100) - 0.5;
            //cout << outputLayerWeights[i][j] << " ";
        }
       // cout << endl;
    }
}

void FeedForwardNetwork::train(const vector < vector< double > > & x, 
const vector< double> &y, size_t numEpochs) {

    size_t trainingexamples = x.size();

   for (size_t epoch = 0; epoch < numEpochs; epoch++) {
       double total_train_loss = 0;
       //print
        cout << "epoch = " << epoch << ", outputs =" << endl;
        for (size_t example = 0; example < trainingexamples; example++) {
            
            // propagate the inputs forward to compute the outputs 
            vector < double > activationInput(inputLayerSize); // store the activation of each of the 784 nodes
            
            // initialize input layer with training data (value of 0-1)
            for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) {
                activationInput[inputNode] = x[example][inputNode];
            }

            vector < double > activationHidden(hiddenLayerSize);
            // calculate activations of hidden layers (for now, just one hidden layer) 
            for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++) {
                double inputToHidden = 0; 
                for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) {
                    inputToHidden += hiddenLayerWeights[inputNode][hiddenNode] * activationInput[inputNode];
                }
                activationHidden[hiddenNode] = g(inputToHidden);
            }

            // 10 output nodes
            vector < double > activationOutput(outputLayerSize);
            // calculate activations of output layer 
            for (size_t outputNode = 0; outputNode < outputLayerSize; outputNode++) {
                double inputAtOutput = 0;
                for(size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++) {
                    inputAtOutput += outputLayerWeights[hiddenNode][outputNode] * activationHidden[hiddenNode];
                }
                activationOutput[outputNode] = g(inputAtOutput);
            }

             cout << example << ": " << setprecision(2) << activationOutput[y[example]] << " " << y[example] << endl;
            
            // calculate the loss
            double loss = 0; 
            for (size_t outputNode = 0; outputNode < 0; outputNode++) {
                loss += pow((activationOutput[outputNode] - y[outputNode]), 2.0);
            }
            total_train_loss += loss;
            // 

            // calculate error of outputnodes
            vector < double > errorOfOutputNodes(10);
            for(size_t outputNode = 0; outputNode < 10; outputNode++) {
                if (y[example] != outputNode) {
                    errorOfOutputNodes[outputNode] = gprime(activationOutput[outputNode]) * (0 - activationOutput[outputNode]);
                } else {
                    errorOfOutputNodes[outputNode] = gprime(activationOutput[outputNode]) * (y[example] - activationOutput[outputNode]);
                }
            }

            // calculate error of hidden layers and also adjusting weights of output layer
            vector< double > errorOfHiddenNode(hiddenLayerSize);
            for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
            {
                for (size_t outputNode = 0; outputNode < 10; outputNode++) {
                    errorOfHiddenNode[hiddenNode] += (outputLayerWeights[hiddenNode][outputNode] * errorOfOutputNodes[outputNode]);
                }
                errorOfHiddenNode[hiddenNode] *= gprime(activationHidden[hiddenNode]);
            }

            // update weights
            // update weight at the output layer 
            for (size_t outputNode = 0; outputNode < 10; outputNode++) {
                for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
                {
                    outputLayerWeights[hiddenNode][outputNode] += alpha * activationHidden[hiddenNode] * errorOfOutputNodes[outputNode];
                }
            }
           // update the weights of the hidden layer
           for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++)
			{
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					hiddenLayerWeights[inputNode][hiddenNode] += alpha * activationInput[inputNode] * errorOfHiddenNode[hiddenNode];
				}
			}

        }
        cout << endl;
   }

   return; 
}