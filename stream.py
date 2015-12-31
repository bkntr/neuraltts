import os
import numpy as np
import mmap
import io
import shutil

from collections import OrderedDict
from utils import find_files, file_parts

__author__ = 'Nir'


class StreamSingle(object):
    """
    :type dtype : np.dtype
    """
    MAGIC_NUMBER = np.array(12345, dtype='uint32')
    DTYPE_LEN = 10

    def __repr__(self):
        if self.header_offset is not None:
            rep = '<%s: %s, %s, len: %d>' % (self.__class__.__name__, self.dtype, self.data_shape.tolist(), self.len)
        else:
            rep = '<%s: empty>' % self.__class__.__name__
        return rep

    def __init__(self, path, mode='r'):
        """
        A simple file format for streaming data
        :type path: file path
        :type mode: str : 'r' | 'w' | 'r+'
        :return:
        """
        assert mode in ['r', 'w', 'r+'], 'given mode is %s' % mode

        self.mode = mode

        if self.mode in ['w', 'r+']:
            folder, _, _ = file_parts(path)
            if folder != '' and not os.path.isdir(folder):
                os.makedirs(folder)

        self.fd = open(path, mode)

        self.dtype = None
        self.data_shape = None
        self.header_offset = None
        self.item_size = None
        self.item_size_bytes = None
        self.len = 0

        if self.mode in ['r', 'r+']:
            self.read_header()

    def write_header(self, data):
        """
        :type data: np.array
        :return:
        """
        self.dtype = data.dtype
        self.data_shape = np.array(data.shape, dtype='uint32')

        np.fromstring(str(self.dtype).ljust(self.DTYPE_LEN, '\0'), dtype='uint8').tofile(self.fd)  # data type
        np.array(len(data.shape), dtype='uint32').tofile(self.fd)  # ndims
        self.data_shape.tofile(self.fd)  # dims
        self.MAGIC_NUMBER.tofile(self.fd)

        self.header_offset = self.fd.tell()
        self.item_size = np.prod(self.data_shape)
        self.item_size_bytes = self.dtype.itemsize * self.item_size
        self.len = 0

    def read_header(self):
        self.fd.seek(0, io.SEEK_END)
        file_size = self.fd.tell()
        if file_size > 0:
            self.fd.seek(0, io.SEEK_SET)
            self.dtype = np.dtype(np.fromfile(self.fd, dtype='uint8', count=self.DTYPE_LEN).tostring().strip('\0'))
            ndims = np.fromfile(self.fd, dtype='uint32', count=1)
            self.data_shape = np.fromfile(self.fd, dtype='uint32', count=ndims)
            magic = np.fromfile(self.fd, dtype='uint32', count=1)
            assert np.all(magic == self.MAGIC_NUMBER)

            self.header_offset = self.fd.tell()
            self.item_size = np.prod(self.data_shape)
            self.item_size_bytes = self.dtype.itemsize * self.item_size

            self.len = (file_size - self.header_offset) / self.item_size_bytes
        else:
            self.len = 0

    def __getitem__(self, item):
        assert isinstance(item, int)

        if self.header_offset is None:
            self.read_header()

        if item >= self.len:
            raise IndexError()

        offset = self.item_size_bytes * item + self.header_offset
        self.fd.seek(offset, io.SEEK_SET)
        data = np.fromfile(self.fd, dtype=self.dtype, count=self.item_size)
        data = data.reshape(self.data_shape)

        return data

    def __setitem__(self, item, data):
        """
        if item is None - append data to file
        :type item: int | None
        :type data: np.array
        """

        assert item is None or isinstance(item, int)
        assert self.mode in ['w', 'r+'], 'mode is set to: %s' % self.mode

        if self.header_offset is None:
            self.write_header(data)
        else:
            assert np.all(data.shape == self.data_shape), \
                'new shape %s, prev shape %s' % (data.shape, self.data_shape)
            assert data.dtype == self.dtype, \
                'new dtype %s, prev dtype %s' % (data.dtype, self.dtype)

        if item is None:
            self.fd.seek(0, io.SEEK_END)

        elif item >= self.len:
            raise IndexError('index %d is out of bound (len: %d)' % (item, self.len))

        else:
            offset = self.item_size_bytes * item + self.header_offset
            self.fd.seek(offset, io.SEEK_SET)

        data.tofile(self.fd)
        if item is None:
            self.len += 1

    def append(self, data):
        """
        :type data: np.array
        """
        self.__setitem__(None, data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return self.len

    def close(self):
        if self.fd:
            self.fd.close()


class Stream(object):
    def __init__(self, folder, mode='r'):
        self.folder = folder
        self.streams = dict()
        self.mode = mode
        self.len = 0

        if self.mode == 'w':
            for f in find_files(self.folder, '*.stream', recursive=False):
                os.unlink(f)
        else:
            self.load_streams()

    def load_streams(self):
        for f in find_files(self.folder, '*.stream', recursive=False):
            _, basename, _ = file_parts(f)
            stream = StreamSingle(f, self.mode)
            if len(self.streams) == 0:
                self.len = len(stream)
            else:
                assert self.len == len(stream), 'length of streams don\'t match (expected %d, %s len is %d)' % (
                    self.len, basename, len(stream))
            self.streams[basename] = stream

    def __getitem__(self, item):
        assert isinstance(item, int)

        if item >= self.len:
            raise IndexError('%d is out of bound (len: %d)' % (item, self.len))

        out = dict()
        for name, stream in self.streams.iteritems():
            out[name] = stream[item]

        return out

    def __setitem__(self, item, data):
        assert isinstance(data, dict)
        assert item is None or isinstance(item, int)

        if len(self.streams) == 0:
            if self.mode in ['r+']:
                self.load_streams()
            else:
                for name in data:
                    self.streams[name] = StreamSingle(os.path.join(self.folder, name + '.stream'), self.mode)

        data_keys = set(data.keys())
        stream_keys = set(self.streams.keys())
        missing_in_data = stream_keys.difference(data_keys)
        missing_in_streams = data_keys.difference(stream_keys)

        if len(missing_in_data):
            raise KeyError('data (%s) is missing the following keys %s' % (list(data_keys), list(missing_in_data)))
        if len(missing_in_streams):
            raise KeyError(
                'stream (%s) is missing the following keys %s' % (list(stream_keys), list(missing_in_streams)))

        for name, value in data.iteritems():
            self.streams[name][item] = value

        if item is None:
            self.len += 1

    def append(self, data):
        self.__setitem__(None, data)

    def __len__(self):
        return self.len

    def keys(self):
        return self.streams.keys()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for s in self.streams.itervalues():
            s.close()

    def __repr__(self):
        rep = ('<%s: \'%s\', len: %s>' % (self.__class__.__name__, self.folder, str(len(self))))
        if len(self.streams) > 0:
            for k, s in self.streams.iteritems():
                rep += '\n\t%s: %s' % (k, s)

        return rep


def main():
    with Stream('test', mode='r+') as s:
        print s
        d = {
            'a': np.ones((3, 2, 15), dtype='float32')*80,
            'b': np.ones((1, 3), dtype='int8')
        }
        s.append(d)
        print(s)

    with Stream('test', mode='r') as s:
        for i in s:
            for k, v in i.iteritems():
                print(k, np.unique(v))
    exit(0)

    with StreamSingle('test.stream', mode='w') as s:
        print s
        d = np.ones((3, 2, 15), dtype='float32') * 3
        s.append(d)
        d2 = np.ones((3, 2, 15), dtype='float32') * 4
        s.append(d2)
        print(s)

    with StreamSingle('test.stream', mode='r+') as s:
        d = np.ones((3, 2, 15), dtype='float32') * 8
        s.append(d)
        for i in s:
            print i.shape, np.unique(i)
        print(s)


if __name__ == '__main__':
    main()
