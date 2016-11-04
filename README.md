Implemented Multilayer Feedforward Neural Network (MLFNN) with Backpropagation (BP) learning.The aim was to code a complete handwritten digit recognizer and test it on the MNIST dataset.

DATASET:
	The MNIST database of handwritten digits,	available from this page, has a training set of 60,000 examples, and a test set of 10,000   		examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
	The dataset and its description can be downloaded from:http://yann.lecun.com/exdb/mnist.

IMPLEMENTATION:
	Recognizer first read the image data, extract features from it and use a multilayer feedforward neural network classifier to recognize 		any test image.
	Number of hidden layer nodes are 90.This was obtained by trial-error and cross validation method.
	File "inputweights90.txt" Contains weights of nodes from input to hidden layers,whereas file "hiddenweights90.txt" weights of nodes 	from hidden to output layers.
	Also implemented K-NN to compare the results of the two methods
RESULTS:
	Neural NW:	94.12
	1NN:		96.33
	5-fold:	95.35
