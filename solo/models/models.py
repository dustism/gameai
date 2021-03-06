import tensorflow.contrib.layers as layers
import tensorflow as tf

def mlp(inpt, n_output, scope, hiddens = [], activation = tf.nn.relu):
	with tf.variable_scope(scope):
		out = inpt
		for hidden in hiddens:
			out = layers.fully_connected(out, num_outputs = hidden, activation_fn = None)
			out = activation(out)

		out = layers.fully_connected(out, num_outputs = n_output, activation_fn = None)

	return out

