"""attention layer"""

import tensorflow as tf


def attention(inputs, attention_size):

    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    hidden_size = inputs.shape[2].value

    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.01))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.01))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.01))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu')
    alphas = tf.nn.softmax(vu, name='alphas')
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return output
