import tensorflow as tf
import tensorflow.contrib.slim as slim

LEN_W2V = 100
MAX_LEN_SENT = 100
CONV_WINDOW = 5
LAYERS = 5

def main():
    with tf.name_scope("input_embedding"):
        question_vector = tf.placeholder(tf.float32, shape=[None, LEN_W2V * MAX_LEN_SENT, 1])
        right_answer_vector = tf.placeholder(tf.float32, shape=[None, LEN_W2V * MAX_LEN_SENT, 1])
        wrong_answer_vector = tf.placeholder(tf.float32, shape=[None, LEN_W2V * MAX_LEN_SENT, 1])

    return

def model(vector):
    with tf.name_scope('conv1') as scope:
        kernel = weight_variable(shape=[CONV_WINDOW, 1, 32])
        conv = tf.nn.conv1d(vector, kernel, [1, 2, 1, 1], padding='SAME')
        biases = bias_variable(shape=[32])
        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope)
        h_pool = max_pool_2x1(conv)

    print 'stacking more layers...'
    for i in xrange(1, LAYERS):
        with tf.name_scope('conv1') as scope:
            kernel = weight_variable(shape=[CONV_WINDOW, 32, 32])
            conv = tf.nn.conv1d(h_pool, kernel, [1, 2, 1, 1], padding='SAME')
            biases = bias_variable(shape=[32])
            bias = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(bias, name=scope)
            h_pool = max_pool_2x1(conv)

    print 'stacking fully connected...'
    W_fc1 = weight_variable([7 * 7 * 32, 128])
    b_fc1 = bias_variable([128])

    h_pool2_flat = tf.reshape(h_pool, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    return h_fc1


def conv2d(x, W):
    return tf.nn.conv1d(x, W, padding='SAME')

def max_pool_2x1(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="weights", trainable=True)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="biases", trainable=True)

if __name__ == '__main__':
    main()