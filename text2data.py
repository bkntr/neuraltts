#!/usr/bin/env python

import subprocess
import re
import h5py
import numpy as np
import progressbar
from fuel.datasets import H5PYDataset

WAV_HEADER_LEN = 44
BATCH = 200
N = 200000

WAV_MAX = 102400
WORD_MAX = 20

def text2wav(text):
    cmd = 'espeak --stdout "%s" 2>>/dev/null' % text
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    return np.fromstring(output, dtype=np.uint8)[WAV_HEADER_LEN:]

words_path = 'data/eng_wikipedia_2010_1M-words.txt'
output_path = 'data/words.hdf5'

pattern = re.compile('[\W_]+')

f = h5py.File(output_path, mode='w')

wavs_dset = f.create_dataset('wavs', (N, WAV_MAX), compression='lzf', dtype=np.uint8)
wavs_dset.dims[0].label = 'batch'
wavs_dset.dims[1].label = 'width'

words_dset = f.create_dataset('words', (N,), compression='lzf', dtype='S20')
words_dset.dims[0].label = 'batch'

n_train = round(N * 0.9)
split_dict = {
    'train': {'wavs': (0, n_train), 'words': (0, n_train)},
    'test': {'wavs': (n_train, N), 'words': (n_train, N)}
}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

bar = progressbar.ProgressBar(maxval=N, widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]).start()

with open(words_path) as fwords:
    words = np.zeros((BATCH,), dtype='S20')
    wavs = np.zeros((BATCH, WAV_MAX), dtype=np.uint8)

    i = 0
    for word in fwords:
        word = word.split('\t')[1]
        word = pattern.sub('', word)
        if not (len(word) > 1 and len(word) <= WORD_MAX):
            continue

        wav = text2wav(word)

        if len(wav) > WAV_MAX:
            continue

        words[i % BATCH] = word
        wavs[i % BATCH, 0:len(wav)] = wav

        i += 1
        if (i % BATCH) == 0:
            begin = i - BATCH
            end = i
            wavs_dset[begin:end, :] = wavs
            words_dset[begin:end] = words

            words = np.zeros((BATCH,), dtype='S20')
            wavs = np.zeros((BATCH, WAV_MAX), dtype=np.uint8)

        bar.update(i)

        if i == N:
            break

bar.finish()
f.flush()
f.close()
