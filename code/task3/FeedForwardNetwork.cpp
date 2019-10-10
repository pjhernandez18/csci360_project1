//  Author by: Paul Hernandez
//  CS 360
//  Date: 10/4/2019
#include "FeedForwardNetwork.h"
#include <iostream> 
#include <fstream> 
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
            hiddenLayerWeights[i][j] = (double) ((rand() % 100) * 1.0 / 100) - 0.5;
        }
    }

    innerhiddenLayerWeights.resize(hiddenLayerSize);
    for (size_t i = 0; i < hiddenLayerSize; i++) {
        innerhiddenLayerWeights[i].resize(hiddenLayerSize);
        for (size_t j = 0; j < hiddenLayerSize; j++) {
            
            innerhiddenLayerWeights[i][j] = (double) ((rand() % 100) * 1.0 / 100) - 0.5;
            
        }
        
    }
    outputLayerWeights.resize(hiddenLayerSize);
    for (size_t i = 0; i < hiddenLayerSize; i++) {
        outputLayerWeights[i].resize(outputLayerSize);
        for (size_t j = 0; j < outputLayerSize; j++) {
    
            outputLayerWeights[i][j] = (double) ((rand() % 100) * 1.0 / 100) - 0.5;
            
        }
       
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
            // calculate activations of hidden layers 
            for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++) {
                double inputToHidden = 0; 
                for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) {
                    inputToHidden += hiddenLayerWeights[inputNode][hiddenNode] * activationInput[inputNode];
                }
                activationHidden[hiddenNode] = g(inputToHidden);
            }

            //calculate activations for second hidden layer
             vector < double > activationHidden2(hiddenLayerSize);
            for (size_t i = 0; i < hiddenLayerSize; i++) {
                double hiddenToHidden = 0; 
                for (size_t j= 0; j < hiddenLayerSize; j++) {
                    hiddenToHidden += innerhiddenLayerWeights[i][j] * activationHidden[i];
                }
                activationHidden2[i] = g(hiddenToHidden);
            }

            // 10 output nodes
            vector < double > activationOutput(outputLayerSize);
            // calculate activations of output layer 
            for (size_t outputNode = 0; outputNode < outputLayerSize; outputNode++) {
                double inputAtOutput = 0;
                for(size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++) {
                    inputAtOutput += outputLayerWeights[hiddenNode][outputNode] * activationHidden2[hiddenNode];
                }
                activationOutput[outputNode] = g(inputAtOutput);
            }
            
            
            // calculate the loss
            double loss = 0; 
            for (size_t outputNode = 0; outputNode < 10; outputNode++) {
                if (y[example] != outputNode) {
                    loss += pow((0 - activationOutput[outputNode]), 2.0);
                } else {
                    loss += pow((1.0 - activationOutput[outputNode]), 2.0);
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
                    errorOfOutputNodes[outputNode] = gprime(activationOutput[outputNode]) * (1.0 - activationOutput[outputNode]);
                }
            }
            // calculate error of hidden layers and also adjusting weights of output layer
            vector< double > errorOfHiddenNode2(hiddenLayerSize);
            for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
            {
                for (size_t outputNode = 0; outputNode < 10; outputNode++) {
                    errorOfHiddenNode2[hiddenNode] += (outputLayerWeights[hiddenNode][outputNode] * errorOfOutputNodes[outputNode]);
                }
                errorOfHiddenNode2[hiddenNode] *= gprime(activationHidden2[hiddenNode]);
            }

            vector< double > errorOfHiddenNode(hiddenLayerSize);
            for (size_t i = 0; i < hiddenLayerSize; i++)
            {
                for (size_t j = 0; j < hiddenLayerSize; j++) {
                    errorOfHiddenNode[i] += (innerhiddenLayerWeights[i][j] * errorOfHiddenNode2[j]);
                }
                errorOfHiddenNode[i] *= gprime(activationHidden[i]);
            }

            // update weights
            // update weight at the output layer 
                for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
                {
                    for (size_t outputNode = 0; outputNode < 10; outputNode++) {
                        outputLayerWeights[hiddenNode][outputNode] += alpha * activationHidden2[hiddenNode] * errorOfOutputNodes[outputNode];
                    }
                }
            // update weight at the second hidden layer
            for (size_t i = 0; i < hiddenLayerSize; i++) {
                for (size_t j = 0; j < hiddenLayerSize; j++)
                {
                    innerhiddenLayerWeights[i][j] += alpha * activationHidden[i] * errorOfHiddenNode2[j];
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
           
    
           if(y[example] == maxOutput) {
               total_correct_train_samples += 1;
           }
           total_train_samples += 1;
        }
       
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
            // calculate activations of hidden layers 
            for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++) {
                double inputToHidden = 0; 
                for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) {
                    inputToHidden += hiddenLayerWeights[inputNode][hiddenNode] * activationInput[inputNode];
                }
                activationHidden[hiddenNode] = g(inputToHidden);
            }

            //calculate activations for second hidden layer
             vector < double > activationHidden2(hiddenLayerSize);
            for (size_t i = 0; i < hiddenLayerSize; i++) {
                double hiddenToHidden = 0; 
                for (size_t j= 0; j < hiddenLayerSize; j++) {
                    hiddenToHidden += innerhiddenLayerWeights[i][j] * activationHidden[i];
                }
                activationHidden2[i] = g(hiddenToHidden);
            }

            // 10 output nodes
            vector < double > activationOutput(outputLayerSize);
            // calculate activations of output layer 
            for (size_t outputNode = 0; outputNode < outputLayerSize; outputNode++) {
                double inputAtOutput = 0;
                for(size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++) {
                    inputAtOutput += outputLayerWeights[hiddenNode][outputNode] * activationHidden2[hiddenNode];
                }
                activationOutput[outputNode] = g(inputAtOutput);
            }
         
             // Calculate predicted label by getting max of activation output
            double maxPredictedLabel = max_element(activationOutput.begin(), activationOutput.end()) - activationOutput.begin();
            
            // Calculate total validation loss
            for (size_t outputNode = 0; outputNode < 10; outputNode++) { 
                if (y[example] != outputNode) {
                    total_val_loss += pow((0 - activationOutput[outputNode]), 2.0);
                } else {
                    total_val_loss += pow((1.0 - activationOutput[outputNode]), 2.0);
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

        train_sample(x , y, total_train_samples, total_train_loss, total_correct_train_samples);
    
        double total_val_samples = 0;
        double total_correct_val_samples = 0;
        double total_val_loss = 0;

        val_sample(w, z, total_val_samples, total_val_loss, total_correct_val_samples);

        // print and plot
        double train_accuracy = total_correct_train_samples / total_train_samples * 100;
        double val_accuracy = total_correct_val_samples / total_val_samples * 100;       
        
        cout << epoch << ", " << train_accuracy << ", " << val_accuracy 
        << ", " << total_train_loss << ", " << total_val_loss << endl;

        
   }


   return; 
}



