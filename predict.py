import sys

import lasagne
import numpy as np
import theano
from fuel.datasets import H5PYDataset
from theano import tensor as T

from network import build_network, iterate_minibatches

HEADER = np.array((0x52, 0x49, 0x46, 0x46, 0x4A, 0xB1, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20,
          0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x22, 0x56, 0x00, 0x00, 0x44, 0xAC, 0x00, 0x00,
          0x02, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61, 0x00, 0xF0, 0xFF, 0x7F), dtype=np.uint8).view(np.uint16)


def predict(snapshot):
    test_set = H5PYDataset('data/words.hdf5', which_sets=('test',), load_in_memory=True, subset=slice(0,5))

    # build network
    network, input_var = build_network()

    # load parameters
    with np.load(snapshot) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # compile prediction function
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_fn = theano.function([input_var], [test_prediction])

    i = 0
    for inputs, _ in iterate_minibatches(test_set, 1, shuffle=True):
        wav = np.round(pred_fn(inputs)[0] * np.iinfo(np.uint16).max)
	with open('wavs/' + str(i) + '.wav', 'wb') as f:
	    np.concatenate((HEADER, wav[0, :].astype(np.uint16))).tofile(f)
        i += 1


if __name__ == '__main__':
    predict('snapshot_33000.npz')
