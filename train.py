#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from fuel.datasets import H5PYDataset

from network import build_network, iterate_minibatches

BATCH = 32
PRINT_INTERVAL = 100
SNAPSHOT_INTERVAL = 1000


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=10):
    target_var = T.matrix('target')

    # Create neural network model
    print('Building model and compiling functions...')
    network, input_var = build_network()

    regularization = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2) * 1e-4

    # loss
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean() + regularization

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean() + regularization

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], test_loss)

    # Finally, launch the training loop.
    print("Starting training...")
    print("# of parameters: ", lasagne.layers.count_params(network))

    print_err = 0

    # Load the dataset
    print('Loading test set...')
    test_set = H5PYDataset('data/words.hdf5', which_sets=('test',), load_in_memory=True)

    # We iterate over epochs:
    for epoch in range(num_epochs):
        for i in range(5):
            print('Loading train set...')
            train_set = H5PYDataset('data/words.hdf5',
                                    which_sets=('train',),
                                    subset=slice(i * 100000, min(i * 100000 + 100000, 450000)),
                                    load_in_memory=True)
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for inputs, targets in iterate_minibatches(train_set, BATCH, shuffle=True):
                err = train_fn(inputs, targets)
                train_err += err
                train_batches += 1

                print_err += err
                if train_batches % PRINT_INTERVAL == 0:
                    print("[{}] loss:\t\t{:.6f}".format(train_batches, print_err / PRINT_INTERVAL))
                    print_err = 0

                if train_batches % SNAPSHOT_INTERVAL == 0:
                    np.savez('snapshot_{}.npz'.format(train_batches), *lasagne.layers.get_all_param_values(network))

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(test_set, BATCH, shuffle=False):
            inputs, targets = batch
            val_err += val_fn(inputs, targets)
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    # After training, we compute and print the test error:
    test_err = 0
    test_batches = 0
    for inputs, targets in iterate_minibatches(train_set, BATCH, shuffle=False):
        test_err += val_fn(inputs, targets)
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    np.savez('snapshot_{}.npz'.format(train_batches), *lasagne.layers.get_all_param_values(network))


if __name__ == '__main__':
    main(1000)
