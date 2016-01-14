import lasagne
import lasagne.layers as ll

from tabulate import tabulate

WAV_MAX = 102400
WORD_MAX = 20

GRAD_CLIP = 100


def print_network(network):
    net_print = []
    for l in lasagne.layers.get_all_layers(network):
        net_print.append([l.name, 'x'.join(map(str, l.output_shape))])
    print(tabulate(net_print))


def build_dense_autoencoder(input_size, units):
    l_in = ll.InputLayer(shape=(None, input_size), name='input')

    denses = [None] * len(units)
    l_prev = l_in
    for i in range(len(units)):
        name = 'dense_%d' % i
        denses[i] = ll.DenseLayer(l_prev, nonlinearity=None, b=None, num_units=units[i], name=name)
        l_prev = ll.NonlinearityLayer(ll.BiasLayer(denses[i], name=name + '_bias'),
                                      name=name + '_nl')

    for i in reversed(range(len(units))):
        name = 'dense_%d' % i
        l_dense_inv = ll.InverseLayer(l_prev, denses[i], name=name + '_inv')
        l_prev = ll.BiasLayer(l_dense_inv, name=name + '_inv_bias')

    print_network(l_prev)

    return l_prev, l_in.input_var


def build_conv_autoencoder(input_size, channels):
    l_in = ll.InputLayer(shape=(None, input_size), name='input')
    l_dimshuffle = ll.DimshuffleLayer(l_in, (0, 'x', 1))

    convs = []
    l_prev = l_dimshuffle
    for i in range(len(channels)):
        name = 'conv_%d' % i
        convs.append(
            ll.Conv1DLayer(l_prev, channels[i], 7, stride=2, nonlinearity=None, b=None, pad='same',
                           name=name))
        l_prev = ll.NonlinearityLayer(ll.BiasLayer(convs[i - 1], name=name + '_bias'),
                                      name=name + '_nl')

    name = 'conv_lowdim'
    l_lowdim = ll.Conv1DLayer(l_prev, 1, 1, nonlinearity=None, b=None, pad='same', name=name)
    l_prev = ll.NonlinearityLayer(ll.BiasLayer(l_lowdim, name=name + '_bias'),
                                  name=name + '_nl')
    l_prev = ll.InverseLayer(l_prev, l_lowdim, name=name + '_inv')
    l_prev = ll.BiasLayer(l_prev, name=name + '_inv_bias')

    for i in reversed(range(len(channels))):
        name = 'conv_%d' % i
        l_prev = ll.InverseLayer(l_prev, convs[i - 1], name=name + '_inv')
        l_prev = ll.BiasLayer(l_prev, name=name + '_inv_bias')

    l_prev = ll.FlattenLayer(l_prev)
    print_network(l_prev)

    return l_prev, l_in.input_var


def build_mfcc_autoencoder(input_size):
    l_in = ll.InputLayer(shape=(None, input_size), name='input')
    l_out = ll.DimshuffleLayer(l_in, (0, 'x', 1))
    l_out = ll.Conv1DLayer(l_out,  num_filters=400, filter_size=401, stride=200, pad='same', name='conv_enc1')
    l_out = ll.Conv1DLayer(l_out, num_filters=80, filter_size=1, name='conv_enc2')
    l_out = ll.Conv1DLayer(l_out, num_filters=400, filter_size=1, name='conv_dec1')
    l_out = ll.ReshapeLayer(l_out, ([0], 200, input_size*2 / 200), name='reshape')
    l_out = ll.Conv1DLayer(l_out, num_filters=200, filter_size=3, stride=2, pad='same', name='conv_dec2')
    l_out = ll.FlattenLayer(l_out)

    print_network(l_out)

    return l_out, l_in.input_var
