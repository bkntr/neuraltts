#!/usr/bin/env python

from __future__ import print_function

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(dataset,
         snapshot=None,
         num_epochs=10,
         batch_size=32,
         print_interval=100,
         snapshot_interval=1000):
    import time

    import numpy as np
    import theano
    import theano.tensor as T

    import lasagne
    from fuel.datasets import H5PYDataset

    from network import build_network, build_autoencoder

    target_var = T.matrix('target')

    # Create neural network model
    print('Building model and compiling functions...')
    network, input_var = build_autoencoder()

    if snapshot:
        # load parameters
        with np.load(snapshot) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    # regularization = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2) * 1e-4
    regularization = 0

    # train loss
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean() + regularization

    # test loss
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean() + regularization

    # updates
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    # Compile
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    test_fn = theano.function([input_var, target_var], test_loss)

    # Load the dataset
    print('Loading datasets...')
    train_set = H5PYDataset(dataset, which_sets=('train',), load_in_memory=True)
    test_set = H5PYDataset(dataset, which_sets=('test',), load_in_memory=True)

    # Finally, launch the training loop.
    print("Starting training...")
    print("   # of parameters: ", lasagne.layers.count_params(network))

    train_err = 0
    train_batches = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        print_time = time.time()
        print('Training...')
        for inputs, targets in iterate_minibatches(train_set, batch_size, shuffle=True):
            train_err = train_fn(inputs, targets)
            train_batches += 1

            # print loss
            if train_batches % print_interval == 0:
                ms_per_iter = (time.time() - print_time) / print_interval * 1000
                print("[{}] loss:\t\t{:.6f} ({.2f} ms/iter)".format(
                        train_batches, train_err / print_interval, ms_per_iter))
                train_err = 0

            # save snapshot
            if train_batches % snapshot_interval == 0:
                np.savez('snapshot_{}.npz'.format(train_batches), *lasagne.layers.get_all_param_values(network))

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(test_set, batch_size, shuffle=False):
            inputs, targets = batch
            val_err += test_fn(inputs, targets)
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    # After training, we compute and print the test error:
    test_err = 0
    test_batches = 0
    for inputs, targets in iterate_minibatches(train_set, batch_size, shuffle=False):
        test_err += test_fn(inputs, targets)
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    np.savez('snapshot_{}.npz'.format(train_batches), *lasagne.layers.get_all_param_values(network))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('-s', '--snapshot')
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('-p', '--print-interval', type=int)
    parser.add_argument('-n', '--snapshot-interval', type=int)

    args = parser.parse_args()

    main(args.dataset,
         args.snapshot,
         args.epochs,
         args.batch_size,
         args.print_interval,
         args.snapshot_interval)
