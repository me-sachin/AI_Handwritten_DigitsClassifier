from dataloader import *
from visualizer import *

from theano import *

import _pickle
import numpy

def predict(
    dataset="mnist.pkl.gz",
    batch_size = 20):

    ###############################################################
    # Get the data
    ###############################################################

    index = T.lscalar()
    y = T.ivector('y')

    datasets = shared_dataset(dataset)
    test_set_x, test_set_y = datasets[2]


    ###############################################################
    # Get the model
    ###############################################################

    classifier = _pickle.load(open('best_model.pkl', 'rb'))

    predict_model = theano.function(
        inputs = [index],
        outputs = classifier.y_pred,
        givens = {
            classifier.input: test_set_x[index * batch_size : (index+1) * batch_size]})

    test_model = theano.function(
        inputs = [index],
        outputs = classifier.errorRate(y),
        givens = {
            classifier.input: test_set_x[index * batch_size : (index+1) * batch_size],
            y: test_set_y[index * batch_size : (index+1) * batch_size]
        })

    ###############################################################
    # Assess the model against the data
    ###############################################################
    
    print("Assessing the model...")
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    test_losses = [test_model(i) for i in range(n_test_batches)]
    test_score = numpy.mean(test_losses)

    print(
        'The classifier correctly recognized %.1f %% of the %d digits.' % (
        ((1.0 - test_score) * 100),
        (n_test_batches * batch_size)))
    
    print("Some illustrative examples...")
    predicted_values = predict_model(0)
    digits = test_set_x.get_value(borrow=True)
    for i in range(len(predicted_values)):
        plot_image(digits[i], 28, 28)     

predict()
    
