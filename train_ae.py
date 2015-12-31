#!/usr/bin/env python

from __future__ import print_function

import os

from data import iterate_wavchunks, iterate_wavchunks_fft


def save_comparison(path, input, pred, rate):
    import matplotlib.pyplot as plt
    from scipy import fft, arange
    def plot_specgram(w, i, rate):
        """
        Plots a Single-Sided Amplitude Spectrum of y(t)
        """
        Ts = 1.0 / rate  # sampling interval
        t = arange(0, float(len(w))/rate, Ts)  # time vector

        plt.subplot(2, 2, i)
        plt.plot(t, w)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.subplot(2, 2, i+1)
        n = len(w)  # length of the signal
        k = arange(n)
        T = float(n) / rate
        frq = k / T  # two sides frequency range
        frq = frq[range(n / 2)]  # one side frequency range

        Y = fft(w) / n  # fft computing and normalization
        Y = Y[range(n / 2)]

        plt.plot(frq, abs(Y), 'r')  # plotting the spectrum
        plt.xlabel('Freq (Hz)')
        plt.ylabel('|Y(freq)|')

    plt.figure()
    plot_specgram(input, 1, rate)
    plot_specgram(pred, 3, rate)
    plt.savefig(path)
    plt.close()


def main(dataset, snapshot, experiment, batch_size, print_interval, snapshot_interval):
    import time

    import numpy as np
    import theano
    import theano.tensor as T

    import lasagne

    from network import build_autoencoder

    target_var = T.matrix('target')

    # Create neural network model
    print('Building model and compiling functions...')
    network, input_var = build_autoencoder()

    snapshot_dir = os.path.join(experiment, 'snapshots')
    if not os.path.isdir(snapshot_dir):
        os.makedirs(snapshot_dir)
    figure_dir = os.path.join(experiment, 'figures')
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    if snapshot:
        # load parameters
        with np.load(os.path.join(snapshot_dir, 'snapshot_') + snapshot + '.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    regularization = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2) * 1e-4
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
    updates = lasagne.updates.adam(loss, params)

    # Compile
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    pred_fn = theano.function([input_var], prediction)

    # Finally, launch the training loop.
    print("Starting training...")
    print("   # of parameters: ", lasagne.layers.count_params(network))

    train_err = 0

    print_time = time.time()
    print('Training...')
    for train_batches, inputs in enumerate(iterate_wavchunks(dataset, 320, batch_size)):
        train_err += train_fn(inputs, inputs)

        # print loss
        if train_batches % print_interval == 0:
            ms_per_iter = (time.time() - print_time) / print_interval * 1000
            print_time = time.time()
            print("[{}] loss:\t\t{:.6f} ({:.2f} ms/iter)".format(
                    train_batches, train_err / print_interval, ms_per_iter))
            train_err = 0

        # save snapshot
        if train_batches % snapshot_interval == 0:
            pred = pred_fn(inputs)
            snapshot_path = os.path.join(snapshot_dir, 'snapshot_{}.npz'.format(train_batches))
            comparison_path = os.path.join(figure_dir, 'figure_{}.png'.format(train_batches))
            np.savez(snapshot_path, *lasagne.layers.get_all_param_values(network))
            save_comparison(comparison_path, inputs[0, :], pred[0, :], 16000)

            # pred_complex = [complex(pred[0, i], pred[0, i+1]) for i in range(0, pred.shape[1], 2)]
            # input_complex = [complex(inputs[0, i], inputs[0, i+1]) for i in range(0, inputs.shape[1], 2)]
            # save_comparison(comparison_path, np.fft.ifft(input_complex), np.fft.ifft(pred_complex), 16000)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('-s', '--snapshot')
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-e', '--experiment', default='.')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-p', '--print-interval', type=int, default=10)
    parser.add_argument('-n', '--snapshot-interval', type=int, default=1000)

    args = parser.parse_args()

    main(args.dataset,
         args.snapshot,
         os.path.join('experiments', '240_conv'),
         32,
         1000,
         10000)
