#include <vector>
#include <iostream>

#include "MNIST_reader.h"
#include "FeedForwardNetwork.h"


using namespace std;
int main()
{
	string filename = "../MNIST/train-images.idx3-ubyte";
	//load MNIST images
	vector <vector< double> > training_images;
	loadMnistImages(filename, training_images);
	cout << "Number of images: " << training_images.size() << endl;
	cout << "Image size: " << training_images[0].size() << endl;

	// Get first 6000 of training examples
	vector <vector< double> >::const_iterator first = training_images.begin();
	vector <vector< double> >::const_iterator last = training_images.begin() + 10;
 	vector<vector<double> > training_images_set(first, last);

	filename = "../MNIST/train-labels.idx1-ubyte";
	//load MNIST labels
	vector<double> training_labels;
	loadMnistLabels(filename, training_labels);
	cout << "Number of labels: " << training_labels.size() << endl;

	// Get first 6000 of training examples
	vector <double>::const_iterator beg = training_labels.begin();
	vector <double>::const_iterator end = training_labels.begin() + 10;
 	vector<double> training_labels_set(beg, end);

	//hyper parameters
	double alpha = 0.2;   // learning rate
	size_t inputLayerSize = 784;
	size_t hiddenLayerSize = 10;
	size_t numEpochs = 10000;
	size_t outputLayerSize = 10;

	// random seed for the network initialization
	int seed = 0;
	
	FeedForwardNetwork nn(alpha, hiddenLayerSize, inputLayerSize, outputLayerSize);
	nn.init(seed); 
	nn.train(training_images_set, training_labels_set, numEpochs);

	return 0;
}