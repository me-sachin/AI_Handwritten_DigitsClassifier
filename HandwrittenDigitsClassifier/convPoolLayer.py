import numpy

from theano import *
import theano.tensor as T
from theano.tensor.signal import pool

class ConvPoolLayer(object):
    """
        + First, the convolutional layer detects local edges/changes.
        + Then, the max-pooling layer reduces/compresses the space and thus globalizes
            the edges/changes that have been detected by the convolutional layer.
    """

    def __init__(self, rng, input, filter_shape, image_shape, pool_size=(2,2)):
        assert filter_shape[1] == image_shape[1]
                
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(image_shape[2:]) / numpy.prod(pool_size))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(
                    low = - numpy.sqrt(6. / (fan_in + fan_out)),
                    high = numpy.sqrt(6. / (fan_in + fan_out)),
                    size = filter_shape),
                dtype = theano.config.floatX),
            borrow = True)

        self.b = theano.shared(
            numpy.zeros(
                (filter_shape[0],),
                dtype = theano.config.floatX),
            borrow = True)

        conv_out = T.nnet.conv2d(
            input = input,
            filters = self.W,
            input_shape = image_shape,
            filter_shape = filter_shape)

        pooled_out = pool.pool_2d(
            input = conv_out,
            ds = pool_size,
            ignore_border = True)

        self.output = numpy.tanh(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.input = input

        self.params = [self.W, self.b]

        
                

