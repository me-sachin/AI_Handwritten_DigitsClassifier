from convPoolLayer import ConvPoolLayer
from hiddenLayer import HiddenLayer
from logisticRegressionLayer import LogisticRegressionLayer

class CNN_MLP(object):
    """
    + First a couple of convolutional and pooling layers react to local changes and globalize these.
    + Second a hidden layer transforms the data space.
    + Third, a logistic regression layer classifies the data.
    """

    def __init__(self, rng, input, n_batch, n_in, n_featureMaps, n_hidden, n_out):

        #image size is 28 x 28, with 1 feature per pixel

        self.convPoolLayer1 = ConvPoolLayer(
            rng = rng,
            input = input.reshape((n_batch, 1, 28, 28)),
            image_shape = (n_batch, 1, 28, 28),
            filter_shape = (n_featureMaps[0], 1, 5, 5),
            pool_size = (2, 2))

        # image size is 12 x 12, with n_featureMaps[0] features per pixel

        self.convPoolLayer2 = ConvPoolLayer(
            rng = rng,
            input = self.convPoolLayer1.output,
            image_shape = (n_batch, n_featureMaps[0], 12, 12),
            filter_shape = (n_featureMaps[1], n_featureMaps[0], 5, 5),
            pool_size = (2, 2))

        # image size is 4 x 4, with n_featureMaps[1] features per pixel

        self.hiddenLayer = HiddenLayer(
            rng = rng,
            input = self.convPoolLayer2.output.flatten(2),
            n_in = n_featureMaps[1] * 4 * 4,
            n_out = n_hidden)

        self.logRegressionLayer = LogisticRegressionLayer(
            input = self.hiddenLayer.output,
            n_in = n_hidden,
            n_out = n_out)

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood)

        self.errorRate = (
            self.logRegressionLayer.errorRate)

        self.y_pred = (
            self.logRegressionLayer.y_pred)

        self.params = (
            self.convPoolLayer1.params +
            self.convPoolLayer2.params +
            self.hiddenLayer.params +
            self.logRegressionLayer.params)

        self.input = input


    