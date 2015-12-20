import lasagne
import numpy as np

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

WAV_MAX = 102400
WORD_MAX = 20
WORD_CHANNELS = 27

GRAD_CLIP = 100


def build_network():
    l_in = lasagne.layers.InputLayer(shape=(None, WORD_MAX))

    network = lasagne.layers.DenseLayer(l_in, num_units=512)

    network = lasagne.layers.ReshapeLayer(network, ([0], 1, 1, -1))

    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(1, 3), pad='same')

    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(1, 3), pad='same')

    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(1, 3), pad='same')

    network = lasagne.layers.Conv2DLayer(network, num_filters=200, filter_size=(1, 3), pad='same')

    network = lasagne.layers.ReshapeLayer(network, ([0], WAV_MAX))

    # l_in = lasagne.layers.InputLayer(shape=(None, WORD_MAX, WORD_CHANNELS))
    # l_mask = lasagne.layers.InputLayer(shape=(None, WORD_MAX))
    # l_forward = lasagne.layers.RecurrentLayer(
    #     l_in, 1024, mask_input=l_mask, grad_clipping=GRAD_CLIP,
    #     W_in_to_hid=lasagne.init.HeUniform(),
    #     W_hid_to_hid=lasagne.init.HeUniform())
    # l_backward = lasagne.layers.RecurrentLayer(
    #     l_in, 1024, mask_input=l_mask, grad_clipping=GRAD_CLIP,
    #     W_in_to_hid=lasagne.init.HeUniform(),
    #     W_hid_to_hid=lasagne.init.HeUniform(), backwards=True)
    # l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])

    return network, l_in.input_var


def iterate_minibatches(dataset, batchsize, shuffle=False):
    if shuffle:
        scheme = ShuffledScheme(examples=dataset.num_examples,
                                batch_size=batchsize,
                                rng=np.random.RandomState())
    else:
        scheme = SequentialScheme(examples=dataset.num_examples, batch_size=batchsize)

    data_stream = DataStream(dataset=dataset, iteration_scheme=scheme)
    for data in data_stream.get_epoch_iterator():
        words = np.fromstring(data[1].lower(), dtype='uint8')
        words -= ord('a')
        words = words.reshape((len(data[1]), -1))
        wavs = data[0]

        yield words, wavs
