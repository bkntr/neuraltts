import sys

import lasagne
import numpy as np
import theano
from fuel.datasets import H5PYDataset
from theano import tensor as T

from network import build_network, iterate_minibatches

HEADER = np.array((0x52, 0x49, 0x46, 0x46, 0x4A, 0xB1, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20,
          0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x22, 0x56, 0x00, 0x00, 0x44, 0xAC, 0x00, 0x00,
          0x02, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61, 0x00, 0xF0, 0xFF, 0x7F), dtype=np.uint8)


def predict(snapshot):
    test_set = H5PYDataset('words.hdf5', which_sets=('test',), load_in_memory=True)

    # build network
    input_var = T.matrix('input')
    network = build_network(input_var)

    # load parameters
    with np.load(snapshot) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # compile prediction function
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_fn = theano.function([input_var], [test_prediction])

    for inputs, _ in iterate_minibatches(test_set, 10, shuffle=True):
        wav = pred_fn(inputs)[0]
        for i in range(10):
            word = inputs[i].tostring().rstrip('\x00')
            with open('wavs/' + word + '.wav', 'wb') as f:
                np.concatenate((HEADER, wav[i, :].astype(np.uint8))).tofile(f)


if __name__ == '__main__':
    predict('snapshot_141.npz')
