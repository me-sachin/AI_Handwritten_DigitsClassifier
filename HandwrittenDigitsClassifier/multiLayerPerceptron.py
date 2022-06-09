from hiddenLayer import HiddenLayer
from logisticRegressionLayer import LogisticRegressionLayer

from theano import *
import theano.tensor as T

class MultiLayerPerceptron(object):
    """
    + A hidden layer transforms the data space:
        x -> tanh(W1*x + b1)
    + Projects the transformed data-points 'x' onto a set of hyperplanes.
        x -> softmax(W2 * tanh(W1*x + b1) + b2)
    + This projection, hence the distance of the transformed data-points to the hyperplane
        represents the class membership probability.
    + Stochastic gradient descent reshapes the hyperplanes to minimize 
        -log(softmax_y(W2 * tanh(W1*x + b1) + b2).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):

        self.hiddenLayer = HiddenLayer(
            rng = rng,
            input = input,
            n_in = n_in,
            n_out = n_hidden,
            activation = T.tanh)

        self.logRegressionLayer = LogisticRegressionLayer(
            input = self.hiddenLayer.output,
            n_in = n_hidden,
            n_out = n_out)

        self.L1 = (
            abs(self.hiddenLayer.W).sum() +
            abs(self.logRegressionLayer.W).sum())

        self.L2_sqr = (
            (self.hiddenLayer.W**2).sum() +
            (self.hiddenLayer.W**2).sum())

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood)

        self.errorRate = (
            self.logRegressionLayer.errorRate)

        self.y_pred = (
            self.logRegressionLayer.y_pred)

        self.params = (
            self.hiddenLayer.params + self.logRegressionLayer.params)

        self.input = input

        

