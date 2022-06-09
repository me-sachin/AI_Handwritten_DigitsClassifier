import numpy

from theano import *
import theano.tensor as T

class HiddenLayer(object):
    """
        + The hidden layer transforms the data space.
        + The points are projected onto hyperplanes. A non-linear function of their
            distance to the planes forms the coordinates of the points in the new referential.
            x -> tanh(W*x + b)
    """

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """ Defines the layer and initializes its parameters"""

        self.input = input
    
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low = -numpy.sqrt(6 / (n_in + n_out)),
                    high = numpy.sqrt(6 / (n_in + n_out)),
                    size = (n_in, n_out)),
                dtype = theano.config.floatX)
            if activation == T.nnet.sigmoid:
                W_values *= 4
            self.W = theano.shared(
                value = W_values,
                name = 'W',
                borrow = True)
            
        if b is None:
            self.b = theano.shared(
                value = numpy.zeros(
                    (n_out,),
                    dtype = theano.config.floatX),
                name = 'b',
                borrow = True)

        self.params = [self.W, self.b]

        self.output = activation( T.dot( self.input, self.W ) + self.b )        
        

