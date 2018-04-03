import tensorflow as tf
import numpy as np

class Memory():
	def __init__(self, capacity):
		self.capacity = capacity
		self.data = np.zeros(capacity, dtype = object)
		self.index = -1 

	def store(self, experience):
		self.index += 1
		self.data[self.index % self.capacity] = experience

	def sample(self, batch_size):
		sample_index = np.random.choice(min(self.index, self.capacity), size = batch_size)
		return self.data[sample_index]


class DeepQNetwork():
	def __init__(self, n_features, n_actions, sess, model, parent_ai = None, scope = 'camp_?_global_?', learning_rate = 1e-3, 
		n_replace_target = 50, hiddens = [32, 32], decay = 0.99, memory_size = 10000, batch_size = 2000, 
		epsilon_decrement = 0.0001, epsilon_lower = 0.001, learn_start = 1000):
		
		self.n_features = n_features
		self.n_actions = n_actions
		self.n_replace_target = n_replace_target
		self.model = model
		self.hiddens = hiddens
		self.batch_size = batch_size
		self.decay = decay
		self.scope = scope
		self.copies = 0
		
		self.epsilon_lower = epsilon_lower
		self.learn_start = learn_start

		self.learn_step = 0
		self.sess = sess
		self.eval_input = tf.placeholder(tf.float32, shape = [None, n_features], name = 'eval_input')
		self.target_input = tf.placeholder(tf.float32, shape = [None, n_features], name = 'target_input')
		self.actions_selected = tf.placeholder(tf.int32, shape = [None, ], name = 'actions_selected')
		self.done = tf.placeholder(tf.float32, shape = [None, ], name = 'done')
		self.rewards = tf.placeholder(tf.float32, shape = [None, ], name = 'rewards')
		self.decays = tf.placeholder(tf.float32, shape = [None, ], name = 'decay')
		
		with tf.variable_scope(scope):
			self._epsilon = tf.get_variable(name = 'epsilon', dtype = tf.float32, initializer = 1.0)
			self._epsilon_decrement = tf.constant(epsilon_decrement)
			self.update_epsilon = tf.assign(self._epsilon, self._epsilon - self._epsilon_decrement)
			
			self.eval_output = model(inpt = self.eval_input, n_output = n_actions, scope = 'eval_net', hiddens = hiddens)
			self.target_output = tf.stop_gradient(model(inpt = self.target_input, n_output = n_actions, scope = 'target_net', hiddens = hiddens))

		self.eval_output_selected = tf.reduce_sum(self.eval_output * tf.one_hot(self.actions_selected, n_actions), axis = 1)
		self.eval_output_target = self.rewards + self.decays * tf.reduce_max(self.target_output, axis = 1) * (1. - self.done)

		self.loss = tf.reduce_mean(tf.squared_difference(self.eval_output_selected, self.eval_output_target))
		self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

		self.eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/eval_net')
		self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/target_net')

		self.update = [tf.assign(x, y) for x, y in zip(self.target_params, self.eval_params)]
		
		if parent_ai is None:	
			self.sess.run(tf.global_variables_initializer())
			self.memory = Memory(capacity = memory_size)
			self.sess.run(self.update)
		else:
			self._sync = [tf.assign(x, y) for x, y in zip(self.eval_params, parent_ai.eval_params)] + \
			[tf.assign(self._epsilon, parent_ai._epsilon)]
			self.parent_ai = parent_ai
			
	def act(self, s):
		if np.random.uniform() < self.epsilon:
			return np.random.choice(self.n_actions)
		else:
			action_values = self.sess.run(self.eval_output, feed_dict = {
    				self.eval_input : s[np.newaxis, :]
    			})
			return np.argmax(action_values, axis = 1)[0]

	def store(self, exp):
		self.memory.store(exp)

	def learn(self):
		self.learn_step += 1
		if self.memory.index < self.batch_size or self.learn_step < self.learn_start : return
		
		s, a, r, s_next, done, decays = self._process_data(self.memory.sample(self.batch_size))
		
		self.sess.run(self.train, feed_dict = {
				self.eval_input : s, 
				self.actions_selected : a, 
				self.rewards : r,
				self.target_input : s_next,
				self.done : done, 
				self.decays: decays
			})

		if self.learn_step % self.n_replace_target == 0:
			self.sess.run(self.update)

		if self.epsilon > self.epsilon_lower:
			self.sess.run(self.update_epsilon)
		
	def sync(self):
		self.sess.run(self._sync)
		self.learn_step = self.parent_ai.learn_step

	@property
	def epsilon(self):
		return self.sess.run(self._epsilon)
		
	def _process_data(self, batch_data):
		s, a, r, s_next, done, decays = [], [], [], [], [], []
		for i in range(self.batch_size):
			end_state = batch_data[i][-1][3]
			decay = 1.
			later_reward = 0
			for j in reversed(range(batch_data[0].shape[0])):
				s.append(batch_data[i][j][0])
				a.append(batch_data[i][j][1])
				later_reward = self.decay * later_reward + batch_data[i][j][2]
				r.append(later_reward)
				s_next.append(end_state)
				done.append(batch_data[i][j][4])
				decay = decay * self.decay
				decays.append(decay)
				
		return s, a, r, s_next, done, decays
		
	def act_only_copy(self):
		player_copy = DeepQNetwork(n_features = self.n_features, n_actions = self.n_actions, sess = self.sess
		, model = self.model, parent_ai = self, scope = self.scope[0:6] + '_local_' + str(self.copies), hiddens = self.hiddens)
		self.copies += 1
		return player_copy
			
class SupervisedLearningNetwork():
	def __init__(self, n_features, n_actions, sess, model, scope = 'SLNetwork', learning_rate = 1e-3, memory_size = int(1e6), batch_size = int(1e3),
	epsilon_decrement = 1e-3, epsilon_lower = 1e-3, parent_ai = None, hiddens = [128, 128], learn_start = int(1e3)):
	
		self.n_actions = n_actions
		self.input = tf.placeholder(tf.float32, shape = [None, n_features], name = 'input')
		self.labels = tf.placeholder(tf.int32, shape = [None, ], name = 'labels')
		
		self.sess = sess
		self.learn_step = 0
		self.learn_start = learn_start
		self.batch_size = batch_size
		self.epsilon_lower = epsilon_lower

		with tf.variable_scope(scope):
			self._create_epsilon(epsilon_decrement)
		
			self.output = model(inpt = self.input, n_output = n_actions, scope = 'network', hiddens = hiddens)
						
			self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.output, labels = self.labels))
			
			self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
			
		self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/network')
			
		if parent_ai is None:
			self.memory = Memory(capacity = memory_size)
		else:
			self._sync = [tf.assign(x, y) for x, y in zip(self.params, parent_ai.params)] + \
			[tf.assign(self._epsilon, parent_ai._epsilon)]
			self.parent_ai = parent_ai

	def act(self, observation):
		if np.random.uniform() < self.epsilon:
			return np.random.randint(0, self.n_actions)
			
		return np.argmax(self.sess.run(self.output, feed_dict = {self.input : observation[np.newaxis, :]})[0])
		
	def store(self, exp):
		self.memory.store(exp)
		
	def learn(self):
		self.learn_step += 1
		
		if self.memory.index < self.batch_size or self.learn_step < self.learn_start : return
		
		batch_memory = self.memory.sample(self.batch_size)
		
		_, loss = self.sess.run([self.train, self.loss], feed_dict = {
				self.input : [batch_memory[i][0] for i in range(self.batch_size)],
				self.labels : [batch_memory[i][1] for i in range(self.batch_size)]
			})
			
		print('in learn_step {}, the loss is {}'.format(self.learn_step, loss))
			
		if self.epsilon > self.epsilon_lower:
			self.sess.run(self.update_epsilon)
		
	def _create_epsilon(self, epsilon_decrement):
		self._epsilon = tf.get_variable(name = 'epsilon', dtype = tf.float32, initializer = 1.0)
		self._epsilon_decrement = tf.constant(epsilon_decrement)
		self.update_epsilon = tf.assign(self._epsilon, self._epsilon - self._epsilon_decrement)
		
	@property
	def epsilon(self):
		return self.sess.run(self._epsilon)
		
	def sync(self):
		self.sess.run(self._sync)
		self.learn_step = self.parent_ai.learn_step