//  Author by: Paul Hernandez
//  CS 360
//  Date: 10/4/2019

#include <vector>
#include <iostream>

#include "MNIST_reader.h"
#include "FeedForwardNetwork.h"

using namespace std;

int main()
{
	string filename = "../MNIST/train-images.idx3-ubyte";
	//load MNIST images
	vector <vector< double> > training_images_all;
	loadMnistImages(filename, training_images_all);

	filename = "../MNIST/train-labels.idx1-ubyte";
	//load MNIST labels
	vector<double> training_labels_all;
	loadMnistLabels(filename, training_labels_all);
	
	// Parition into training and validation sets
	// training
	vector<vector<double> > training_set(training_images_all.begin(), training_images_all.begin() + 4000);
 	vector<double> training_labels(training_labels_all.begin(), training_labels_all.begin() + 4000);
	
	// validation
	vector<vector<double> > validation_set(training_images_all.begin() + 4000, training_images_all.begin() + 6000);
 	vector<double> validation_labels(training_labels_all.begin() + 4000, training_labels_all.begin() + 6000);

	string fname = "../MNIST/t10k-labels.idx3-ubyte";
	vector <vector< double> > test_images_all;
	loadMnistImages(fname, test_images_all);

	string lname = "../MNIST/t10k-labels.idx1-ubyte";
	vector<double> test_labels_all;
	loadMnistLabels(lname, test_labels_all);
 
	//hyper parameters
	double alpha = 0.1;   // learning rate
	size_t inputLayerSize = 784;
	size_t hiddenLayerSize = 32;
	size_t numEpochs = 10000;
	size_t outputLayerSize = 10;

	// random seed for the network initialization
	int seed = 0;
	
	FeedForwardNetwork nn(alpha, hiddenLayerSize, inputLayerSize, outputLayerSize);
	nn.init(seed);
	nn.train(training_set, training_labels, validation_set, validation_labels, numEpochs);

	// Need a way to obtain optimum Epoch somehow and simulate using test data!
	
	return 0;
}