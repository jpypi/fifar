# Based on scripts at https://github.com/tensorflow/tensorflow/contrib/learn/python/learn/datasets/

'''Dataset utilities'''

import pickle
import collections
from os import path

from tensorflow.python.framework import dtypes
import numpy as np

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def load_cifar10(data_path):
    """Load the CIFAR10 dataset.

    Args:
        data_path: string, path to the folder containing the cifar10 dataset

    Returns:
        Datasets tuple containing the train, validation, and test datasets
    """
    train1 = unpickle(path.join(data_path, 'data_batch_1'))
    train_data = train1[b'data']
    train_target = dense_to_one_hot(train1[b'labels'], 10)

    train2 = unpickle(path.join(data_path, 'data_batch_2'))
    train_data = np.concatenate((train_data, train2[b'data']), axis=0)
    train_target = np.concatenate((train_target, dense_to_one_hot(train2[b'labels'], 10)), axis=0)

    train3 = unpickle(path.join(data_path, 'data_batch_3'))
    train_data = np.concatenate((train_data, train3[b'data']), axis=0)
    train_target = np.concatenate((train_target, dense_to_one_hot(train3[b'labels'], 10)), axis=0)

    train_data = train_data.reshape(-1, 32*32*3)

    train = DataSet(train_data, train_target)

    validate1 = unpickle(path.join(data_path, 'data_batch_4'))
    valid_data = validate1[b'data']
    valid_target = dense_to_one_hot(validate1[b'labels'], 10)

    valid_data = valid_data.reshape(-1, 32*32*3)

    validation = DataSet(valid_data, valid_target)

    test1 = unpickle(path.join(data_path, 'test_batch'))
    test_data = test1[b'data']
    test_target = dense_to_one_hot(test1[b'labels'], 10)

    test_data = test_data.reshape(-1, 32*32*3)

    test = DataSet(test_data, test_target)

    return Datasets(train=train, validation=validation, test=test)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    if type(labels_dense) != np.ndarray:
        labels_dense = np.asarray(labels_dense)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def unpickle(path):
    fo = open(path, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

#Dataset class taken shamelessly from tensorflow's MNIST tutorial files
class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float32,
                 normalize=True,
                 reshape=True):
        """Construct a DataSet.
        'dtype' can either be 'uint8' to leave the input as '[0, 255]', or 'float32'
        to rescale into '[0, 1]'.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                    dtype)

        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]


        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0]
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

        if normalize:
            images = self.preprocess(images)

        # Convert shape from [num_examples, rows*columns*channels] to
        # [num_examples, rows, columns, channels]
        if reshape:
            images = images.reshape(-1, 3, 32, 32).transpose(0,2,3,1)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def preprocess(self,images):
        '''Normalize the data.'''
        sub_mean = np.subtract(images, np.mean(images, axis=0))
        div_std = np.divide(sub_mean, np.std(sub_mean, axis=0))
        return div_std

    def next_batch(self, batch_size, shuffle=True):
        '''Return the next 'batch_size' examples from this data set.'''

        start = self._index_in_epoch
        #Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        #Go to the next epoch
        if start + batch_size > self._num_examples:
            #Finished Epoch
            self._epochs_completed += 1
            #Get ther est of examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            #Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            #Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), \
                   np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

