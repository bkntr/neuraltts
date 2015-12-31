import os
import wave

from data import iterate_wavs


def main(dataset, snapshot, experiment, batch_size, print_interval, snapshot_interval):
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
    # load parameters
    with np.load(os.path.join(snapshot_dir, 'snapshot_') + snapshot + '.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # predict
    prediction = lasagne.layers.get_output(network)
    pred_fn = theano.function([input_var], prediction)

    pred_dir = os.path.join(experiment, 'pred')
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)

    print('Predicting...')
    for train_batches, (f, input) in enumerate(iterate_wavs(dataset, 320, batch_size)):
        wav_file = wave.open(os.path.join(pred_dir, f), 'w')
        wav_file.setparams((1, 2, 16000, len(input), 'NONE', 'not compressed'))

        pred = np.zeros((len(input),), dtype=np.uint16)
        for frame in range(0, len(input), 320):
            input_pad = np.zeros((1, 320), dtype=np.float32)
            input_nopad = input[frame:frame+320]
            input_pad[0, :len(input_nopad)] = input_nopad
            pred[frame:frame+320] = (pred_fn(input_pad) * np.iinfo(np.uint16).max).astype(np.uint16)[0, :len(pred[frame:frame+320])]

        wav_file.writeframes((pred * np.iinfo(np.uint16).max).astype(dtype=np.uint16))
        wav_file.close()


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

    main('/home/benk/uni/shai/neuraltts/data/1919/142785/wavs',
         '50000',
         os.path.join('experiments', '240'),
         args.batch_size,
         args.print_interval,
         args.snapshot_interval)
