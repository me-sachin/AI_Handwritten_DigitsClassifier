from dataloader import *
from CNN_MLP import CNN_MLP
from visualizer import *

from theano import *

import _pickle
import timeit

# Build model
def train_convolutionalNeuralNetwork(
    learning_rate = 0.1,
    n_epochs = 200,
    dataset="mnist.pkl.gz",
    batch_size = 20,
    n_featureMaps = [20, 50],
    n_hidden = 500):

    ###############################################################
    # Get Data
    ###############################################################
     
    # Load datasets
    datasets = shared_dataset(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Visualize some data samples
    plot_image(train_set_x.get_value(borrow=True)[10], 28, 28)
    plot_image(valid_set_x.get_value(borrow=True)[15], 28, 28)
    plot_image(test_set_x.get_value(borrow=True)[5], 28, 28)

    # Split sets into batches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


    ###############################################################
    # Build model
    ###############################################################

    print("Building the model...")

    # Allocate symbolic variables
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    # Random number generator for weight initialization of hidden layer
    rng = numpy.random.RandomState(1234)

    # Build classifier
    classifier = CNN_MLP(
        rng = rng,
        input = x,
        n_batch = batch_size,
        n_in = 28*28,
        n_featureMaps = n_featureMaps,
        n_hidden = n_hidden,
        n_out = 10)

    # Define gradient descent
    cost = (
        classifier.negative_log_likelihood(y))

    g_params = [T.grad(cost=cost, wrt=param) for param in classifier.params]

    updates = [
        (param, param - g_param * learning_rate)
        for param, g_param in zip(classifier.params, g_params)]

    # Test function
    test_model = theano.function(
        inputs = [index],
        outputs = classifier.errorRate(y),
        givens = {
            x: test_set_x[index * batch_size : (index+1) * batch_size],
            y: test_set_y[index * batch_size : (index+1) * batch_size]
        })

    # Validation function
    validate_model = theano.function(
        inputs = [index],
        outputs = classifier.errorRate(y),
        givens = {
            x: valid_set_x[index * batch_size : (index+1) * batch_size],
            y: valid_set_y[index * batch_size : (index+1) * batch_size]
        })

    # Training function
    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: train_set_x[index * batch_size : (index+1) * batch_size],
            y: train_set_y[index * batch_size : (index+1) * batch_size]
        })


    ###############################################################
    # Train Model
    ###############################################################
    
    print("Training the model...")
    patience = 1000 # look at this many batches regardless
    patience_increase = 2   # wait this much longer when a new best is found

    improvement_threshold = 0.995 # a relative improvement of this much is considered significant

    validation_frequency = min(n_train_batches, patience//2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for batch_index in range(n_train_batches):
            batch_avg_cost = train_model(batch_index)

            iter = (epoch - 1) * n_train_batches + batch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, batch %i/%i, validation error rate %f %%' % (
                    epoch,
                    batch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter*patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(
                        '    epoch %i, batch %i/%i, test error rate %f %%' % (
                        epoch,
                        batch_index + 1,
                        n_train_batches,
                        test_score * 100))

                    with open('best_model.pkl', 'wb') as f:
                        _pickle.dump(classifier, f)

            if (patience <= iter):
                done_looping = True
                break

    end_time = timeit.default_timer()

    print((
        'Optimization completed with best validation loss of %f %%,'
        'obtained at iteration %i'
        'with test score of %f %%.') % (
        best_validation_loss * 100.,
        best_iter,
        test_score * 100.))

    print(
        'The code ran for %d epochs, withiin %f minutes.' % (
        epoch,
        (end_time - start_time)/60.))
        
train_convolutionalNeuralNetwork()