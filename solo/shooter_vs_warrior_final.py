import sys
if __name__ == '__main__':
	sys.path.append('src')
	sys.path.append('models')
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
import pickle

def play(ip, port, self_ai, global_ai, dataQ, scoreQ):
	env = LOLEnv(ip, port)

	episode = 0

	history = History(HISTORY_LENGTH)

	evaluation_set = []
	set_got = False
	max_episodes = MAX_EPISODES // 5 if port == EVALUATION_PORT else MAX_EPISODES
	
	while NEW_EPISODE:
		episode += 1
		frame_counter = 0

		synchronize_version(self_ai, global_ai)
		time_start = time.time()

		_, observation = env.reset()
		feature = extract_feature(observation)
		while NEXT_FRAME:

			if frame_counter % 2 == 0:
				if port != EVALUATION_PORT:
					action = self_ai.act(feature)
				else:
					action = self_ai.act_greedy(feature)
				action_id, action_wrapped = wrap_action(action, observation)
			else:
				action_id, action_wrapped = enemy_make_decision(observation)

			observation_next = consume_frame(SKIP_FRAME, env, action_id, action_wrapped)

			feature_next = extract_feature(observation_next)

			winner = judge_winner(observation_next)

			reward = shape_reward(observation, observation_next, winner, frame_counter)
	
			if frame_counter % 2 == 0 and action_wrapped is not None:
				done = 1. if winner is not None else 0.
				if port != EVALUATION_PORT:
					history.put((feature, action, reward, feature_next, done))
					if history.full() : dataQ.put(history.get())
				elif SAVE_STATES and not set_got:
					evaluation_set.append(feature)

			if winner is not None or frame_counter >= max_episodes:
				time_end = time.time()
				print('Episode {} end, time : {}, frame : {}'.format(episode, time_end - time_start, frame_counter))
				if winner:
					if winner == CAMP_RED:
						print('CAMP_RED win!')
					elif winner == CAMP_BLUE:
						print('CAMP_BLUE win!')
				else:
					print('No winner...')

				if port == EVALUATION_PORT:
					score = health_difference(observation) / 1000.
					print('health difference: %f' % score)
					scoreQ.put(score)
					if SAVE_STATES and not set_got:
						with open(EVALUATION_FILE_PATH, 'wb') as f:
							pickle.dump(np.array(evaluation_set), f, pickle.HIGHEST_PROTOCOL)
							print('pickle saved.')
						set_got = True

				print('epsilon : %f' % self_ai.epsilon)
				env.end()
				break

			observation = observation_next
			feature = feature_next

			frame_counter += 1


if __name__ == '__main__':

	Sess = tf.Session()
	global_ai = DeepQNetwork(n_features = QUANTITY_FEATURES, n_actions = QUANTITY_ACTIONS, 
		scope = 'global_ai', model = mlp, parent_ai = None, sess = Sess, 
		learning_rate = 5e-3, n_replace_target = 50, hiddens = [64, 64, 64, 64], decay = 0.99, memory_size = 100000, batch_size = 2000, 
		epsilon_decrement = 1e-3, epsilon_lower = 0.001, learn_start = LEARN_START)
	dataQ = queue.Queue()

	score_plotter = ScorePlotter()
	scoreQ = queue.Queue()
	qvalue_plotter = QValuePlotter()
	with open(EVALUATION_FILE_PATH, 'rb') as f:
	    evaluation_set = pickle.load(f)

	ais = []
	for i in range(N_GAME):
		ais.append(DeepQNetwork(n_features = QUANTITY_FEATURES, n_actions = QUANTITY_ACTIONS,
		scope = 'local_ai_' + str(i), model = mlp, parent_ai = global_ai, sess = Sess,
		hiddens = [64, 64, 64, 64]))

	Saver = tf.train.Saver()

	if RESTORE:
		Saver.restore(Sess, RESTORE_PATH)
		print('restored successfully from ' + RESTORE_PATH)
		
	else:
		Sess.run(tf.global_variables_initializer())

	for i in range(N_GAME):		
		new_thread = td.Thread(target = play, args = (IP, PORT[i], ais[i], global_ai, dataQ, scoreQ))
		new_thread.start()

	while LEARNING:
		evaluate_qvalue(global_ai, qvalue_plotter, evaluation_set)
		plot_score(score_plotter, scoreQ)
		fetch_data(global_ai, dataQ)
		global_ai.learn()
		if not ONLY_PLAY and global_ai.learn_step % SAVE_EVERY == 0:
			save_path = Saver.save(Sess, SAVE_PATH + str(global_ai.learn_step) + '.ckpt')
			print('saved in' + save_path)
