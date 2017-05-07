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

    train_data = preprocess_fn(train_data)

    train = DataSet(train_data, train_target)

    validate1 = unpickle(path.join(data_path, 'data_batch_4'))
    valid_data = validate1[b'data']
    valid_target = dense_to_one_hot(validate1[b'labels'], 10)

    valid_data = preprocess_fn(valid_data)

    validation = DataSet(valid_data, valid_target)

    test1 = unpickle(path.join(data_path, 'test_batch'))
    test_data = test1[b'data']
    test_target = dense_to_one_hot(test1[b'labels'], 10)

    test_data = preprocess_fn(test_data)

    test = DataSet(test_data, test_target)

    return Datasets(train=train, validation=validation, test=test)

def preprocess_fn(image_array):
    '''Turn a single, contiguous array of image data into a
    [num_samples, 32, 32, 3] numpy array,
    each pixel mean centered around 0'''
    #Scale to [0,1]
    images = np.multiply(image_array, 1.0 / 255.0)

    images = images - np.mean(images, axis=0)

    images = images.reshape(-1, 3, 32, 32)

    #Get the per-pixel means (across channels)
    #means = np.mean(images, axis=1)
    #Dim is [-1, 32, 32], expand so we can broadcast
    #means = np.expand_dims(means, axis=1)

    #images = images - means

    return images.transpose(0,2,3,1)


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
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='bytes')


#Dataset class taken shamelessly from tensorflow's MNIST tutorial files
class DataSet(object):

    def __init__(self,
                 images,
                 labels):
        """Construct a DataSet."""

        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

        self._num_examples = images.shape[0]

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

