#include <stdio.h>
#include <math.h>
#define g(x) (1.0/(1.0+exp(-x)))
#define gprime(x) (g(x)*(1-g(x))) 

int main() 
{ 
	float alpha = 10.0;   // learning rate
	int trainingexamples = 4; 
	int features = 2; 
	float x[4][2] = {	{0, 0},
								{0, 1}, 
								{1, 0}, 
								{1, 1}}; 
	float y[4] = {0, 1, 1, 0}; 

	// initialize the network
	float w[3] = {1.1, -2.1, 0.3}; 	// random values

	// train the network
	for (int epoch = 0; 1; ++epoch) 
	{
 		for (int l = 0; l < trainingexamples; ++l) 
		{
 			float weightedsum = 0.0; 
			for (int j = 0; j < features; ++j) 
				weightedsum += w[j]* x[l][j];
			for (int j = 0; j < features; ++j)
 				w[j] -= alpha*(g(weightedsum)- y[l])*gprime(weightedsum)*x[l][j];
		} 
		printf("epoch = %d, weights =", epoch); 
		for (int j = 0; j < features; ++j) 
			printf(" %.2f", w[j]); 
		printf (", outputs ="); 
		for (int l = 0; l < trainingexamples; ++l) 
		{ 
			float weightedsum = 0.0;
			for (int j = 0; j < features; ++j) 
				weightedsum += w[j]*x[l][j]; 
			printf(" %.2f", g(weightedsum)); 
		} 
		printf("\n"); 
	} 
}