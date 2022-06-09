# Motivation

I believe that modern interfaces to computers, like touch, gesture, sound and vision, empower
humans to use computers more effectively (do the right stuff) and efficiently (do the stuff right).
Imagine a world in which we can express ourselves in a natural way, and computers could still 
understand us.

One step in that direction is the ability for computers to read our handwriting and translate it into
digitized ASCII characters. To start easy, I restrict the focus on classifying single handwritten 
digits, from 0 to 9.


# Procedure

## The Data

Since everyone's handwriting is slightly different, I needed a broad data set of handwritten digits,
written by different people. The [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
of handwritten digits suited my needs and contains tens of thousands of 28x28 grayscale images of
handwritten digits.

I split the data set as follows:
* 50'000 training images
* 10'000 validation images
* 10'000 test images
	
## The Classifier

In this project, I followed a [deep learning tutorial](http://deeplearning.net/tutorial/) to
implement a deep network (a simplified version of LeNet5), formed by the following layer stack:

| Layer                          | Synapses                                | Neurons                                           |
|:-------------------------------|:----------------------------------------|:--------------------------------------------------|
| Input Image Batch              | 20 images with 28 x 28 pixels per image |	                                               |
| Convolutional Layer            | 20 features with 5 x 5 field of view    | 20 features for each of the 24 x 24 sub-images    |
| MaxPooling Layer               | 2 x 2 compression                       | 20 features for each of the 12 x 12 sub-images    |
| Convolutional Layer            | 50 features with 5 x 5 field of view    | 50 features for each of the 8 x 8 sub-images      |
| MaxPooling Layer               | 2 x 2 compression                       | 50 features for each of the 4 x 4 sub-images      |
| Multilayer Perceptron Layer    | 800 x 500 space transformation          | 500 hidden features in the hidden layer           |
| Logistic Regression Classifier | 500 x 10 classification                 | The probability of the image to be a 0, 1, 2, … 9 |

* The convolutional layers encode patterns (mainly edges) in the image.
* The maxpooling layers compress the image and indicate whether that specific sub-region contains the feature.
* The multilayer perceptron transforms the data space.
* The Logistic Regression Layer finally computes the classification probabilities.

## The Training

The classifier is trained via stochastic gradient descent (see the code for the learning rates and hyper-parameters)
on the test set. The model is occasionally evaluated on the validation set, and if a new lowest error-rate is hit on
the validation set, the model is saved.

## The Tools

The classifier has been coded in Python (v3.6.1) and makes extensive use of the [Theano framework](http://www.deeplearning.net/software/theano/) (v0.8.2).

## The Code

The code Is available on gitHub.
You still need to download the dataset and place the file mnist.pkl.gz within the code directory.

The constitutive layers can be found in the files:
* LogisticRegressionLayer.py
* HiddenLayer.py
* ConvPoolLayer.py
	
These allow you to build the following classifiers:
* LogisticRegressionLayer.py
* MultiLayerPerceptron.py
* CNN_MLP.py
	
To train the model of your choice, run the respective file:
* Train_LogisticRegression.py
* Train_MultiLayerPerceptron.py
* Train_CNN_MLP.py
	
To see how your classifier performs, you may run:
* Predict.py
	
A few helper files are used too:
* Dataloader.py
* Visualizer.py


# Results

The trained convolutional network multilayer perceptron correctly classifies 99.0% of the 10'000
digits in the test set. This number could be slightly higher, were I patient enough to let the
training run longer.

Without the convolutional network, the multilayer perceptron correctly classifies about 97% and
the logistic regression classifier, without the hidden layer, classifies about 93% of the digits 
correctly.