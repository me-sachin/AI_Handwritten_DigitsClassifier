import gzip
import numpy
import pickle
from theano import *
import theano.tensor as T

def load_dataset(filename):
    """Grabs the training-, validation- and test-datasets from disk"""
    with gzip.open(filename, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set


def shared_variables(data_xy, data_usage):
    """Splits a dataset (training, validation or test) into its X and Y parts, as shared variables"""
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))

    data_r, data_c = shared_x.get_value(borrow=True, return_internal_type=True).shape
    print(data_usage + " set (" + str(data_r) + ", " + str(data_c) + ") loaded.")

    return shared_x, T.cast(shared_y, 'int32')


def shared_dataset(filename):
    """Returns the training-, validation- and test-datasets as shared variables"""
    train_set, valid_set, test_set = load_dataset(filename)

    train_x, train_y = shared_variables(train_set, 'Training')
    valid_x, valid_y = shared_variables(valid_set, 'Validation')
    test_x, test_y = shared_variables(test_set, 'Test')

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
