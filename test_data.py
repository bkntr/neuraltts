from scipy import fft, ifft, arange
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
import numpy as np

from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

# dset = H5PYDataset('words.hdf5', which_sets=('test',))
# scheme = ShuffledScheme(examples=dset.num_examples, batch_size=10, rng=numpy.random.RandomState())
#
# data_stream = DataStream(dataset=dset, iteration_scheme=scheme)
# for data in data_stream.get_epoch_iterator():
#     for wav, word in zip(data[0], data[1]):
#         with open(word + '.wav', 'w') as f:
#             f.write(wav)
#     break
#     # print ', '.join([s for s in data[1]])

"""
"""

def plot_specgram(w, rate):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    Ts = 1.0 / rate  # sampling interval
    t = arange(0, float(len(w))/rate, Ts)  # time vector

    plt.subplot(2, 1, 1)
    plt.plot(t, w)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    n = len(w)  # length of the signal
    k = arange(n)
    T = float(n) / rate
    frq = k / T  # two sides frequency range
    frq = frq[range(n / 2)]  # one side frequency range

    Y = fft(w) / n  # fft computing and normalization
    Y = Y[range(n / 2)]

    plt.plot(frq, abs(Y), 'r')  # plotting the spectrum
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')

rate, data = wave.read('/home/benk/uni_ubuntu/shai/data/segmented/wavn/CA-BB-01-02.wav')
data_fft = fft(data)
data_fft = abs(data_fft)
data_ifft = ifft(data_fft)
wave.write('/tmp/test.wav', rate, data_ifft.astype(np.int16))
# plot_specgram(data[100:500].astype(np.float), rate)
# plt.show()
