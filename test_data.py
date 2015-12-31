import numpy
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

dset = H5PYDataset('words.hdf5', which_sets=('test',))
scheme = ShuffledScheme(examples=dset.num_examples, batch_size=10, rng=numpy.random.RandomState())

data_stream = DataStream(dataset=dset, iteration_scheme=scheme)
for data in data_stream.get_epoch_iterator():
    for wav, word in zip(data[0], data[1]):
        with open(word + '.wav', 'w') as f:
            f.write(wav)
    break
    # print ', '.join([s for s in data[1]])

"""
"""

