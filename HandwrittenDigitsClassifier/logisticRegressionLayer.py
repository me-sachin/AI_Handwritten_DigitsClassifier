from theano import *
import theano.tensor as T

class LogisticRegressionLayer(object):
    """
    + Projects data-points 'x' onto a set of hyperplanes 'W*xb'.
    + The distance to these hypeplanes expresses the class membership probability, quantified as
         softmax_i(W*x+b) = e^{W_i*x+b_i} / sum_j{e^{W_j*x+b_j}
    + Stochastic gradient descent reshapes the hyperplanes to minimize -log(softmax_y(W*x+b)), and hence
         the distance from point x to its projection W_i*x+b_i is maximized.
    """

    def __init__(self, input, n_in, n_out):
        """ Defines the model and initializes its parameters"""

        self.input = input

        # Parameters of the model
        self.W = theano.shared(
            value = numpy.zeros(
                (n_in, n_out),
                dtype = theano.config.floatX),
            name = 'W',
            borrow = True)

        self.b = theano.shared(
            value = numpy.zeros(
                (n_out,),
                dtype = theano.config.floatX),
            name = 'b',
            borrow = True)

        self.params = [self.W, self.b]

        # Expression for class membership probability
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # Expression for class memebership prediction
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        """Computes the mean of the negative log-likelihood of the prediction 
        of this model under a given target distribution."""
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errorRate(self, y):
        """Returns the rate of incorrectly classified samples"""
        return T.mean(T.neq(self.y_pred, y))