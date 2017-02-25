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
    return tf.nn.conv2d(x, W, srides=[1,1,1,1], padding="VALID")

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding="VALID")


image_size = 32*32
classes = 10

x  = tf.placeholder(tf.float32, shape=[None, image_size])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#x_image = tf.reshape(x, [-1, 32, 32, 3])

# Reduce image size to 29x29
W_conv_1 = weight_variable([4, 4, 3, 64])
b_conv_1 = bias_variable([64])
conv_1 = tf.nn.elu(conv2d(,W_conv_1) + b_conv_1)

# Reduce image size to 26x26
W_conv_2 = weight_variable([4, 4, 64, 96])
b_conv_2 = bias_variable([96])
conv_2 = tf.nn.elu(conv2d(conv1, W_conv_2) + b_conv_2)

# Reduce image size to 13x13
pool_1 = max_pool_3x3(conv_2)

# Reduce image size to 10x10
W_conv_3 = weight_variable([4, 4, 96, 48])
b_conv_3 = bias_variable([48])
conv_3 = tf.nn.elu(conv2d(pool1, W_conv_3) + b_conv_3)

# Reduce image size to 5x5
pool_2 = max_pool_3x3(conv_3)
pool_2_flat = tf.reshape(pool_2, [-1, 1200])

W_full_1  = weight_variable([1200, 800])
b_full_1 = bias_variable([800])
full_1 = tf.nn.elu(tf.matmul(pool_2_flat, W_full_1) + b_full_1)

W_full_2 = weight_variable([800, 10])
b_full_2 = bias_variable([10])
out = tf.nn.elu(tf.matmul(full_1, W_full_2) + b_full_2)

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))

# Compute gradients, compute parameter changes, and update parameters
train_step = tf.train.Adadelta().minimize()

# The session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


start = time.time()

#for i in range():
#    if i%100 == 0:
#        train_accuracy =
#    sess.run(train_step, feed_dict={x : batch_x
#                                    y_: batch_y,
#                                    keep_prob: 0.5})

print(time.time() - start)

# Test here
