import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *

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


# def build_autoencoder():
#     l_in = InputLayer(shape=(None, 320), name='input')
#     l_dense1 = DenseLayer(l_in, nonlinearity=None, b=None, num_units=240, name='dense1')
#     l_dense1_nl = NonlinearityLayer(lasagne.layers.BiasLayer(l_dense1, name='dense1_bias'), name='dense1_nl')
#
#     l_dense2 = DenseLayer(l_dense1_nl, nonlinearity=None, b=None, num_units=160, name='dense2')
#     l_dense2_nl = NonlinearityLayer(lasagne.layers.BiasLayer(l_dense2, name='dense2_bias'), name='dense2_nl')
#
#     l_dense3 = DenseLayer(l_dense2_nl, nonlinearity=None, b=None, num_units=80, name='dense3')
#     l_dense3_nl = NonlinearityLayer(lasagne.layers.BiasLayer(l_dense3, name='dense3_bias'), name='dense3_nl')
#     l_dense3_inv = InverseLayer(l_dense3_nl, l_dense3, name='dense3_inv')
#     l_dense3_inv_bias = BiasLayer(l_dense3_inv, name='dense3_inv_bias')
#
#     l_dense2_inv = InverseLayer(l_dense3_inv_bias, l_dense2, name='dense2_inv')
#     l_dense2_inv_bias = BiasLayer(l_dense2_inv, name='dense2_inv_bias')
#
#     l_dense1_inv = InverseLayer(l_dense2_inv_bias, l_dense1, name='dense1_inv')
#     l_dense1_inv_bias = BiasLayer(l_dense1_inv, name='dense1_inv_bias')
#
#     return l_dense1_inv_bias, l_in.input_var

# def build_autoencoder(depth):
#     units = 320
#     l_in = InputLayer(shape=(None, units), name='input')
#
#     denses = []
#     l_prev = l_in
#     for i in range(1, depth+1):
#         name = 'dense_%d' % i
#         denses.append(DenseLayer(l_prev, nonlinearity=None, b=None, num_units=units / 2**i, name=name))
#         l_dense_nl = NonlinearityLayer(BiasLayer(denses[i-1], name=name + '_bias'),
#                                        name=name + '_nl')
#         l_prev = l_dense_nl
#
#     for i in reversed(range(1, depth+1)):
#         name = 'dense_%d' % i
#         l_dense_inv = InverseLayer(l_prev, denses[i-1], name=name + '_inv')
#         l_prev = BiasLayer(l_dense_inv, name=name + '_inv_bias')
#
#     print_network(l_prev)
#
#     return l_prev, l_in.input_var

def build_autoencoder(depth):
    units = 320
    l_in = InputLayer(shape=(None, units), name='input')
    l_dimshuffle = DimshuffleLayer(l_in, (0, 'x', 1))

    convs = []
    l_prev = l_dimshuffle
    for i in range(1, depth+1):
        name = 'conv_%d' % i
        convs.append(Conv1DLayer(l_prev, 4*2**i, 11, stride=2, nonlinearity=None, b=None, pad='same', name=name))
        l_prev = NonlinearityLayer(BiasLayer(convs[i-1], name=name + '_bias'),
                                   name=name + '_nl')

    name = 'conv_lowdim'
    l_lowdim = Conv1DLayer(l_prev, 1, 3, nonlinearity=None, b=None, pad='same', name=name)
    l_prev = NonlinearityLayer(BiasLayer(l_lowdim, name=name + '_bias'),
                                   name=name + '_nl')
    l_prev = InverseLayer(l_prev, l_lowdim, name=name + '_inv')
    l_prev = BiasLayer(l_prev, name=name + '_inv_bias')

    for i in reversed(range(1, depth+1)):
        name = 'conv_%d' % i
        l_prev = InverseLayer(l_prev, convs[i-1], name=name + '_inv')
        l_prev = BiasLayer(l_prev, name=name + '_inv_bias')

    l_prev = FlattenLayer(l_prev)
    print_network(l_prev)

    return l_prev, l_in.input_var

