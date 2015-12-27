import lasagne
import numpy as np

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from tabulate import tabulate

WAV_MAX = 102400
WORD_MAX = 20

_vals = 'abcdefghijklmnopqrstuvwxyz0123456789\x00'
_max_val = np.max([ord(c) for c in _vals])
_lut = np.zeros((_max_val + 1, 1), dtype=np.uint8)
for i in range(len(_vals)):
    _lut[ord(_vals[i])] = i

WORD_CHANNELS = len(_vals)
EPSILON = WORD_CHANNELS - 1

GRAD_CLIP = 100


def build_network():
    # l_in = lasagne.layers.InputLayer(shape=(None, WORD_CHANNELS, WORD_MAX))
    #
    # network = lasagne.layers.DenseLayer(l_in, num_units=512)
    #
    # network = lasagne.layers.ReshapeLayer(network, ([0], 1, 1, -1))
    #
    # network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(1, 3), pad='same')
    #
    # network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(1, 3), pad='same')
    #
    # network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(1, 3), pad='same')
    #
    # network = lasagne.layers.Conv2DLayer(network, num_filters=200, filter_size=(1, 3), pad='same')

    # network = lasagne.layers.ReshapeLayer(network, ([0], WAV_MAX))

    l_in = lasagne.layers.InputLayer(shape=(None, WORD_CHANNELS, WORD_MAX), name='input')
    l_upscale1 = lasagne.layers.Upscale1DLayer(l_in, 2, name='upscale1')
    l_conv1 = lasagne.layers.Conv1DLayer(l_upscale1, num_filters=64, filter_size=5, pad='same', name='conv1')
    l_upscale2 = lasagne.layers.Upscale1DLayer(l_conv1, 2, name='upscale2')
    l_conv2 = lasagne.layers.Conv1DLayer(l_upscale2, num_filters=64, filter_size=5, pad='same', name='conv2')
    l_upscale3 = lasagne.layers.Upscale1DLayer(l_conv2, 2, name='upscale3')
    l_conv3 = lasagne.layers.Conv1DLayer(l_upscale3, num_filters=64, filter_size=5, pad='same', name='conv3')
    l_shuffle1 = lasagne.layers.DimshuffleLayer(l_conv3, (0, 2, 1), name='shuffle1')
    l_forward = lasagne.layers.RecurrentLayer(
        l_shuffle1, 250, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(), name='forward')
    l_backward = lasagne.layers.RecurrentLayer(
        l_shuffle1, 250, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(), backwards=True, name='backward')
    l_add = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward], name='add')
    l_shuffle2 = lasagne.layers.DimshuffleLayer(l_add, (0, 2, 1), name='shuffle2')
    l_out = lasagne.layers.FlattenLayer(l_shuffle2, name='flatten')

    net_print = []
    for l in lasagne.layers.get_all_layers(l_out):
        net_print.append([l.name, 'x'.join(map(str, l.output_shape))])
    print(tabulate(net_print))

    return l_out, l_in.input_var


def iterate_minibatches(dataset, batchsize, shuffle=False):

    if shuffle:
        scheme = ShuffledScheme(examples=dataset.num_examples,
                                batch_size=batchsize,
                                rng=np.random.RandomState())
    else:
        scheme = SequentialScheme(examples=dataset.num_examples, batch_size=batchsize)
    data_stream = DataStream(dataset=dataset, iteration_scheme=scheme)
    for data in data_stream.get_epoch_iterator():
        words = np.take(_lut, np.fromstring(np.char.lower(data[1]), dtype='uint8').reshape((len(data[1]), -1)))
        words = np.eye(len(_vals), dtype=np.float32)[words].transpose(0, 2, 1)
        wavs = data[0].view(np.uint16).astype(np.float32) / np.iinfo('uint16').max

        yield words, wavs
