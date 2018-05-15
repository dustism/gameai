import tensorflow.contrib.layers as layers
import tensorflow as tf


def mlp(inputs, n_output, scope, hiddens, activation=tf.nn.relu, dueling=True):

    with tf.variable_scope(scope):
        out = inputs
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            out = activation(out)

        if dueling:
            advantage = layers.fully_connected(out, num_outputs=n_output, activation_fn=None)
            value = layers.fully_connected(out, num_outputs=1, activation_fn=None)
            out = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        else:
            out = layers.fully_connected(out, num_outputs=n_output, activation_fn=None)

    return out
