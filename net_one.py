#!/usr/bin/env python3

from input_data import load_cifar10

import time
import numpy as np
import tensorflow as tf

# The structure is
# conv->conv->pool->conv->pool->full->full(out)
# 4x4 patch stride 1 convolutions, 3x3 stride 2 max pooling
# We use elu units, Adadelta optimization


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


image_size = 32*32
classes = 10

x  = tf.placeholder(tf.float32, shape=[None, 3*1024])
y_ = tf.placeholder(tf.int64, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 32, 32, 3])

# Reduce image size to 29x29
W_conv_1 = weight_variable([4, 4, 3, 64])
b_conv_1 = bias_variable([64])
conv_1 = tf.nn.elu(conv_2d(x_image, W_conv_1) + b_conv_1)

# Reduce image size to 26x26
W_conv_2 = weight_variable([4, 4, 64, 96])
b_conv_2 = bias_variable([96])
conv_2 = tf.nn.elu(conv_2d(conv_1, W_conv_2) + b_conv_2)

# Reduce image size to 12x12
pool_1 = max_pool_3x3(conv_2)

# Reduce image size to 9x9
W_conv_3 = weight_variable([4, 4, 96, 48])
b_conv_3 = bias_variable([48])
conv_3 = tf.nn.elu(conv_2d(pool_1, W_conv_3) + b_conv_3)

## Reduce image size to 4x4
#pool_2 = max_pool_3x3(conv_3)
#pool_2_flat = tf.reshape(pool_2, [-1, 4*4*48])
pool_2_flat = tf.reshape(conv_3, [-1, 9*9*48])

W_full_1  = weight_variable([9*9*48, 600])
b_full_1 = bias_variable([600])
full_1 = tf.nn.elu(tf.matmul(pool_2_flat, W_full_1) + b_full_1)
full_1_drop = tf.nn.dropout(full_1, keep_prob)

W_full_2 = weight_variable([600, 10])
b_full_2 = bias_variable([10])
out = tf.nn.elu(tf.matmul(full_1_drop, W_full_2) + b_full_2)


cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))

is_correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

# Compute gradients, compute parameter changes, and update parameters
#train_step = tf.train.AdadeltaOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

# The session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


start = time.time()


path = "cifar-10"
train, valid, test = load_cifar10(path)

# This can be used to find the new dimms after various layers
#print(pool_2.get_shape())

while train.epochs_completed < 10000:
    curr_epoch = train.epochs_completed
    while curr_epoch == train.epochs_completed:
        batch_x, batch_y = train.next_batch(200, True)
        sess.run(train_step, feed_dict={x : batch_x, y_: batch_y, keep_prob: 0.5})
    valid_x, valid_y = valid.next_batch(1000,False)
    train_accuracy = sess.run(accuracy, feed_dict={x : valid_x,
                                                   y_: valid_y,
                                                   keep_prob: 1.0})
    print("Epoch %d: %g"%(train.epochs_completed, train_accuracy))


print(time.time() - start)
