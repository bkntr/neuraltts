import os
from random import randint

import numpy as np
import scipy.io.wavfile as wave

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream


_vals = 'abcdefghijklmnopqrstuvwxyz0123456789\x00'
_max_val = np.max([ord(c) for c in _vals])
_lut = np.zeros((_max_val + 1, 1), dtype=np.uint8)
for i in range(len(_vals)):
    _lut[ord(_vals[i])] = i

WORD_CHANNELS = len(_vals)
EPSILON = WORD_CHANNELS - 1


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


def iterate_wavchunks(root, chunk_size, batch_size):
    wavs = []
    for f in os.listdir(root):
        rate, data = wave.read(os.path.join(root, f))
        wavs.append(data)
        if len(wavs) >= 1000:
            break

    assert wavs

    while True:
        w = wavs[randint(0, len(wavs)-1)]
        batch = np.zeros((batch_size, chunk_size), dtype=np.float32)
        for i in range(batch_size):
            chunk_start = randint(0, len(w) - chunk_size - 1)
            batch[i, :] = w[chunk_start:(chunk_start + chunk_size)]
        #yield batch / batch_max, batch_max
        yield batch


def iterate_wavchunks_fft(root, chunk_size, batch_size):
    wavs = []
    for f in os.listdir(root):
        rate, data = wave.read(os.path.join(root, f))
        wavs.append(data.astype(np.uint16))

    assert wavs

    while True:
        w = wavs[randint(0, len(wavs)-1)].astype(np.float32) / np.iinfo(np.uint16).max
        batch = np.zeros((batch_size, chunk_size * 2), dtype=np.float32)
        for i in range(batch_size):
            chunk_start = randint(0, len(w) - chunk_size - 1)
            fft = np.fft.fft(w[chunk_start:(chunk_start + chunk_size)])
            batch[i, :] = np.vstack((fft.real, fft.imag)).T.flatten()
        batch_max = batch.max()
        yield batch / batch_max, batch_max


def iterate_wavs(root):
    for f in os.listdir(root):
        rate, data = wave.read(os.path.join(root, f))
        yield f, data.astype(np.float32)
