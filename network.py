import lasagne
import theano.tensor as T

from tabulate import tabulate
from data import WORD_CHANNELS

WAV_MAX = 102400
WORD_MAX = 20

GRAD_CLIP = 100


def print_network(network):
    net_print = []
    for l in lasagne.layers.get_all_layers(network):
        net_print.append([l.name, 'x'.join(map(str, l.output_shape))])
    print(tabulate(net_print))


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

    return l_out, l_in.input_var


def build_autoencoder():
    l_in = lasagne.layers.InputLayer(shape=(None, 320), name='input')
    l_dense1 = lasagne.layers.DenseLayer(l_in, nonlinearity=None, b=None, num_units=240)
    l_nl1 = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_dense1))
    l_inv1 = lasagne.layers.InverseLayer(l_nl1, l_dense1)
    l_bias1 = lasagne.layers.BiasLayer(l_inv1)
    l_shuffle = lasagne.layers.DimshuffleLayer(l_bias1, (0, 'x', 1))
    l_conv = lasagne.layers.Conv1DLayer(l_shuffle, 1, 11, pad='same', nonlinearity=lasagne.nonlinearities.linear)
    l_out = lasagne.layers.FlattenLayer(l_conv)

    return l_out, l_in.input_var

