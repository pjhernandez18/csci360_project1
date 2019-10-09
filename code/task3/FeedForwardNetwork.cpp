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
            hiddenLayerWeights[i][j] = (double) ((rand() % 200) * 1.0 / 100) - 1.0;
           // cout << hiddenLayerWeights[i][j] << " ";
        }
        //cout << endl;
    }
    outputLayerWeights.resize(hiddenLayerSize);
    for (size_t i = 0; i < hiddenLayerSize; i++) {
        outputLayerWeights[i].resize(outputLayerSize);
        for (size_t j = 0; j < outputLayerSize; j++) {
            // initialized random weights from -0.5 to 0.5
            outputLayerWeights[i][j] = (double) ((rand() % 200) * 1.0 / 100) - 1.0;
            //cout << outputLayerWeights[i][j] << " ";
        }
       // cout << endl;  
    }
}

void FeedForwardNetwork::train_sample(const vector < vector< double > > & x, 
const vector< double> &y, double &total_train_samples, double &total_train_loss, double &total_correct_train_samples) {

      for (size_t example = 0; example < x.size(); example++) {
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
            //cout << example << ": " << setprecision(2) << activationOutput[y[example]] << " " << y[example] << " ";
            
            // calculate the loss
            double loss = 0; 
            for (size_t outputNode = 0; outputNode < 10; outputNode++) {
                if (y[example] != outputNode) {
                    loss += pow((activationOutput[outputNode] - 0), 2.0);
                } else {
                    loss += pow((activationOutput[outputNode] - y[outputNode]), 2.0);
                }
            }
            total_train_loss += loss;

            // begin backward propagation
            // calculate error of outputnodes
            vector < double > errorOfOutputNodes(outputLayerSize);
            for(size_t outputNode = 0; outputNode < 10; outputNode++) {
                if (y[example] != outputNode) {
                    errorOfOutputNodes[outputNode] = gprime(activationOutput[outputNode]) * (0 - activationOutput[outputNode]); 
                } else {
                    errorOfOutputNodes[outputNode] = gprime(activationOutput[outputNode]) * (y[example] - activationOutput[outputNode]);
                }
            }
            // calculate error of hidden layers (for now, one) and also adjusting weights of output layer
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
           for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++){
                for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
                {
                    hiddenLayerWeights[inputNode][hiddenNode] += alpha * activationInput[inputNode] * errorOfHiddenNode[hiddenNode];
                }
           }
           double maxOutput = max_element(activationOutput.begin(), activationOutput.end()) - activationOutput.begin();
           //cout << "maxIndex: " << maxOutput << endl;
    
           if(y[example] == maxOutput) {
               total_correct_train_samples += 1;
           }
           total_train_samples += 1;
        }
       // cout << endl;
}

void FeedForwardNetwork::val_sample(const vector < vector< double > > & x, 
const vector< double> &y, double &total_val_samples, double &total_val_loss, double &total_correct_val_samples) {
        
    for (size_t example = 0; example < x.size(); example++) {
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
             //cout << example << ": " << setprecision(2) << activationOutput[y[example]] << " " << y[example] << " ";

             // Calculate predicted label by getting max of activation output
            double maxPredictedLabel = max_element(activationOutput.begin(), activationOutput.end()) - activationOutput.begin();
            // cout << "maxPredictedLabel: " << maxPredictedLabel << endl;

            // Calculate total validation loss
            for (size_t outputNode = 0; outputNode < 10; outputNode++) { 
                if (y[example] != outputNode) {
                    total_val_loss += pow((activationOutput[outputNode] - 0), 2.0);
                } else {
                    total_val_loss += pow((activationOutput[outputNode] - y[outputNode]), 2.0);
                }
            }
            if(y[example] == maxPredictedLabel) {
                total_correct_val_samples += 1;
            }
            total_val_samples += 1;
    }

}

void FeedForwardNetwork::train(const vector < vector< double > > & x, 
const vector< double> &y, const vector < vector< double > > & w, 
const vector< double> &z, size_t numEpochs) { 

   for (size_t epoch = 0; epoch < numEpochs; epoch++) {
       double total_train_samples = 0;
       double total_train_loss = 0;
       double total_correct_train_samples = 0;
       //print
       // cout << "Training: " << endl;
        //cout << "epoch = " << epoch << ", outputs =" << endl;
        train_sample(x , y, total_train_samples, total_train_loss, total_correct_train_samples);
        cout << endl;

        double total_val_samples = 0;
        double total_correct_val_samples = 0;
        double total_val_loss = 0;

        //cout << "Validation: " << endl;
        val_sample(w, z, total_val_samples, total_val_loss, total_correct_val_samples);

        // print and plot
        double train_accuracy = total_correct_train_samples / total_train_samples * 100;
        double val_accuracy = total_correct_val_samples / total_val_samples * 100;
        double train_loss = total_train_loss / total_train_samples;
        double val_loss = total_val_loss / total_val_samples;

        cout << "epoch = " << epoch << " Accuracy: " << train_accuracy << " " << val_accuracy 
        << " Loss: " << train_loss << " " << val_loss << endl;

        // if (train_accuracy >= 90 || val_accuracy >= 90) {
        //     break;
        // }
   }
    
   return; 
}



