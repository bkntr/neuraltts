import os

import scipy.io.wavfile as wave
from data import iterate_wavs
from train_ae import load_snapshot


CHUNK_SIZE = 200


def main(dataset, snapshot, experiment, batch_size, print_interval, snapshot_interval, depth):
    import numpy as np
    import theano
    import theano.tensor as T

    import lasagne

    import network as net


    # Create neural network model
    print('Building model and compiling functions...')
    network, input_var = net.build_dense_autoencoder(CHUNK_SIZE, depth)

    snapshot_dir = os.path.join(experiment, 'snapshots')
    # load parameters
    load_snapshot(network, snapshot_dir, snapshot)

    # predict
    prediction = lasagne.layers.get_output(network)
    pred_fn = theano.function([input_var], prediction)

    pred_dir = os.path.join(experiment, 'pred')
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)

    print('Predicting...')
    for train_batches, (f, input) in enumerate(iterate_wavs(dataset)):
        pred = np.zeros((len(input),), dtype=np.int16)
        for frame in range(0, len(input) - CHUNK_SIZE, CHUNK_SIZE):
            chunk = input[frame:frame+CHUNK_SIZE].reshape((1, CHUNK_SIZE))
            m, s = chunk.mean(), chunk.std()
            chunk = (chunk - m) / s
            pred[frame:frame+CHUNK_SIZE] = (pred_fn(chunk) * s + m).astype(np.int16)

        wave.write(os.path.join(pred_dir, f), 16000, pred)

        break


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

    main('/home/benk/uni_ubuntu/shai/data/mansfield1_16000/',
         '30000',
         os.path.join('experiments', 'dense/15-01-16_10-53-45'),
         args.batch_size,
         args.print_interval,
         args.snapshot_interval,
         depth=[100, 20])
