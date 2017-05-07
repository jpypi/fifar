#!/usr/bin/env python3

from input_data import load_cifar10

import argparse
import sys
import numpy as np
import tensorflow as tf

# The structure is
# conv->conv->pool->conv->pool->full->full(out)
# 4x4 patch stride 1 convolutions, 3x3 stride 2 max pooling
# We use elu units, Adadelta optimization

FLAGS = None

def train():
    #Import data
    train_data, valid_data, test_data = load_cifar10(FLAGS.data_dir)

    sess = tf.Session()

    # Helpful functions borrowed from the tensorflow tutorial
    def weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias_variable(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def conv_2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="VALID")

    def max_pool_3x3(x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1], padding="VALID")

    def variable_summaries(var):
        '''Attach a lot of summaries to a Tensor for TensorBoard visualization.'''
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.elu):
        '''Make a simple fully connected layer.

        Do matrix multiply, bias add, and nonlinearity.
        It also sets up name scoping so resulting graph is easy to read,
        and adds summary ops.'''

        with tf.name_scope(layer_name):
            #State of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def conv_layer(input_tensor, input_dims, output_dim, layer_name, act=tf.nn.elu):
        '''Make a simple convolution layer.

        Do matrix multiply, convolution, bias add, and nonlinearity.
        It also sets up name scoping so resulting graph is easy to read,
        and adds summary ops.'''

        with tf.name_scope(layer_name):
            #State of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable(input_dims)
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_conv_plus_b'):
                preactivate = conv_2d(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    with tf.name_scope('input'):
        x  = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        y_ = tf.placeholder(tf.int64, shape=[None, 10])

    # Reduce image size to 29x29
    input_dims = [4, 4, 3, 32]
    output_dim = 32
    conv_1 = conv_layer(x, input_dims, output_dim, 'conv_1')

    # Reduce image size to 26x26
    input_dims = [4, 4, 32, 48]
    output_dim = 48
    conv_2 = conv_layer(conv_1, input_dims, output_dim, 'conv_2')

    # Reduce image size to 12x12
    with tf.name_scope('pool_1'):
        pool_1 = max_pool_3x3(conv_2)

    # Reduce image size to 9x9
    input_dims = [4, 4, 48, 24]
    output_dim = 24
    conv_3 = conv_layer(pool_1, input_dims, output_dim, 'conv_3')

    ## Reduce image size to 4x4
    with tf.name_scope('conv_3_reshape'):
        conv_3_flat = tf.reshape(conv_3, [-1, 9*9*24])

    full_1 = nn_layer(conv_3_flat, 9*9*24, 600, 'full_1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        full_1_drop = tf.nn.dropout(full_1, keep_prob)

    out = nn_layer(full_1_drop, 600, 10, 'out', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Compute gradients, compute parameter changes, and update parameters
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate,
                                            epsilon=0.0001).minimize(
                cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    #Merge all summaries and write out to
    # /tmp/tensorflow/cifar10/logs/cifar10_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test/')

    # The session
    sess.run(tf.global_variables_initializer())

    #Train the model, and write summaries
    def feed_dict(train):
        if train:
            xs, ys = train_data.next_batch(100)
            k = FLAGS.dropout
        else:
            xs, ys = valid_data.next_batch(2500)
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}


    for i in range(FLAGS.max_steps):
        if i % 400 == 0: # Record summaries and valid-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Validation accuracy at step %s: %s' % (i, acc))
        else: # Record train set summaries, and train
            if i % 250 == 249: # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            elif i % 100 == 0: # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
            else:
                sess.run([train_step], feed_dict=feed_dict(True))

    train_writer.close()
    test_writer.close()

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='cifar-10',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str,
                        default='/tmp/tensorflow/fifar/logs',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

