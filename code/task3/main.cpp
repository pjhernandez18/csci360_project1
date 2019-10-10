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
	cout << "Number of images: " << training_images_all.size() << endl;
	cout << "Image size: " << training_images_all[0].size() << endl;

	// for(size_t i = 0; i < training_images_all[0].size(); i++) {
	// 	cout << training_images_all[0][i] << " ";
	// }
	// cout << endl; 

	// Get first 6000 of training examples
	vector <vector< double> >::const_iterator first = training_images_all.begin();
	vector <vector< double> >::const_iterator last = training_images_all.begin() + 6000;
 	vector<vector<double> > training_images_set(first, last);

	filename = "../MNIST/train-labels.idx1-ubyte";
	//load MNIST labels
	vector<double> training_labels_all;
	loadMnistLabels(filename, training_labels_all);
	cout << "Number of labels: " << training_labels_all.size() << endl;

	// Get first 6000 of training labels
	vector <double>::const_iterator beg = training_labels_all.begin();
	vector <double>::const_iterator end = training_labels_all.begin() + 6000;
 	vector<double> training_labels_set(beg, end);

	// Parition into training and validation sets
	// training
	vector<vector<double> > training_set(training_images_set.begin(), training_images_set.begin() + 4000);
 	vector<double> training_labels(training_labels_set.begin(), training_labels_set.begin() + 4000);
	
	// validation
	vector<vector<double> > validation_set(training_images_set.begin() + 4000, training_images_set.begin() + 6000);
 	vector<double> validation_labels(training_labels_set.begin() + 4000, training_labels_set.begin() + 6000);

	//hyper parameters
	double alpha = 0.5;   // learning rate
	size_t inputLayerSize = 784;
	size_t hiddenLayerSize = 150;
	size_t numEpochs = 10000;
	size_t outputLayerSize = 10;

	// random seed for the network initialization
	int seed = 0;
	
	FeedForwardNetwork nn(alpha, hiddenLayerSize, inputLayerSize, outputLayerSize);
	nn.init(seed);
	nn.train(training_set, training_labels, validation_set, validation_labels, numEpochs);

	return 0;
}