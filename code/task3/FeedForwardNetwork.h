
//  Author by: Paul Hernandez
//  CS 360
//  Date: 10/4/2019
#pragma once
#include <math.h>
#include <vector> 

using namespace std;

class FeedForwardNetwork {
    public:
        void init(int seed);

        void train(const vector< vector< double> >& x,
        const vector<double>& y, size_t numEpochs);

        FeedForwardNetwork(double alpha, size_t hiddenLayerSize, size_t inputLayerSize, size_t outputLayerSize) : 
        alpha(alpha), hiddenLayerSize(hiddenLayerSize), inputLayerSize(inputLayerSize), outputLayerSize(outputLayerSize) {}

    private: 
        vector< vector< double > > hiddenLayerWeights;
        vector< vector< double > > outputLayerWeights; 

        double alpha;
        size_t hiddenLayerSize; 
        size_t inputLayerSize;
        size_t outputLayerSize; 

        inline double g(double x) {return 1.0/ (1.0 + exp(-x)); }
        inline double gprime(double y) {return y * (1-y); }
};