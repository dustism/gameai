# you can refer to the config.py and defines.py in src folder for some help

import sys
if __name__ == '__main__':
	sys.path.append('../../src')
	sys.path.append('../../models')
from LOLEnv import LOLEnv
from learn import DeepQNetwork
from utility import *
from Defines import *
from config import *
from models import *
import numpy as np
import collections
import threading as td
import math
import time
import tensorflow as tf
import queue
import gym


def play(ip, port, self_ai, global_ai, dataQ):
	env = gym.make('CartPole-v1')
	episode = 0
	total_score = 0

	history = History(HISTORY_LENGTH)

	while True:
		
		t = 0
		episode += 1	

		synchronize_version(self_ai, global_ai)

		time.sleep(0.1)
		observation = env.reset()

		while True:
			# env.render()
			t += 1
			action = self_ai.act(observation)
			observation_next, reward, done, _ = env.step(action)
			if done and t < 500:
				reward = -20
			history.put((observation, action, reward, observation_next, done))
			if t > HISTORY_LENGTH:
				dataQ.put(history.get())
			if done:
				total_score += t
				if episode % 100 == 0:
					print('in episode {}, ai score {}, epsilon {}'.format(episode, total_score / 100, self_ai.epsilon))
					total_score = 0
				break
			observation = observation_next


if __name__ == '__main__':
	Sess = tf.Session()

	global_ai = DeepQNetwork(n_features = 4, n_actions = 2, 
		scope = 'global_ai', model = mlp, parent_ai = None, sess = Sess, 
		learning_rate = 1e-2, n_replace_target = 50, hiddens = [32, 32], decay = 0.99, memory_size = 10000, batch_size = 2000, 
		epsilon_decrement = 1e-3, epsilon_lower = 0.001, learn_start = 0)
	dataQ = queue.Queue()

	ais = []
	for i in range(N_GAME):
		ais.append(DeepQNetwork(n_features = 4, n_actions = 2,
		scope = 'local_ai_' + str(i), model = mlp, parent_ai = global_ai, sess = Sess,
		hiddens = [32, 32]))

	Saver = tf.train.Saver()

	if RESTORE:
		Saver.restore(Sess, RESTORE_PATH)
		print('restored successfully from ' + RESTORE_PATH)
	else:
		Sess.run(tf.global_variables_initializer())

	for i in range(N_GAME):		
		new_thread = td.Thread(target = play, args = (IP, PORT[i], ais[i], global_ai, dataQ))
		new_thread.start()

	while LEARNING:
		fetch_data(global_ai, dataQ)
		global_ai.learn()
#		if global_ai.learn_step % SAVE_EVERY == 0:
#			save_path = Saver.save(Sess, SAVE_PATH + str(global_ai.learn_step) + '.ckpt')
#			print('saved in' + save_path)
