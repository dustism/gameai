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
import os


# 0 FOR BLUE AND 1 FOR RED
THIS_TURN = 0

def play(ip, port, local_ai, global_ai, dataQ):
	
	# ========= initializer ========================
	env = LOLEnv(ip, port)
	episode = 0
	history = History(HISTORY_LENGTH)
	
	while NEW_EPISODE:

		# ===== re-initialize =======================
		episode += 1
		frame_counter = 0
		time_start = time.time()
		history.clear()

		# ===== get global parameters ===============
		synchronize_version(local_ai[0], global_ai[0])
		synchronize_version(local_ai[1], global_ai[1])		

		_, observation = env.reset()
		feature = extract_feature(observation, frame_counter % 2)

		while NEXT_FRAME:

			# =========== ai make decision ==========
			action = local_ai[current_player(frame_counter)].act(feature)
			action_id, action_wrapped = wrap_action(action, observation)


			# =========== get next feature ==========
			observation_next = consume_frame(SKIP_FRAME, env, action_id, action_wrapped)
			feature_next_current_p = extract_feature(observation_next, current_player(frame_counter))
			feature_next_next_p = extract_feature(observation_next, next_player(frame_counter))

			# =========== winner reward and done ====
			winner = judge_winner(observation_next)
			reward = shape_reward(observation, observation_next, winner, current_player(frame_counter))
			done = 1. if winner is not None else 0.


			# ==== same one training and acting =====
			if action_wrapped is not None and current_player(frame_counter) == THIS_TURN:
				history.put((feature, action, reward, feature_next_current_p, done))
				if history.full() : dataQ[THIS_TURN].put(history.get())

			# === is game end or not ================
			if winner is not None or frame_counter >= MAX_EPISODES:
				# ====== time, frames, scores and epsilon ======
				time_end = time.time()
				score = reward if winner is not None else health_difference(observation) / 1000.
				
				print('Episode {} end, time : {}, frame : {}'.format(episode, time_end - time_start, frame_counter))
				print('Score : %f' % score)
				print('epsilon : %f' % self_ai.epsilon)
				
				env.end()
				break

			observation = observation_next
			feature = feature_next_next_p
			frame_counter += 1



if __name__ == '__main__':

	# ============== ensure the folder exists ====
	os.makedirs(SAVE_PATH)

	# ============== some initializers ===========
	Sess = tf.Session()
	global_ai = [0, 0]
	local_ais = [[], [], [], []]
	dataQ = [queue.Queue(), queue.Queue()]
	counter = 0
	

	# ============= 0 for blue and 1 for red=======
	# ============= global ai =====================
	for i in range(2):
		global_ai[i] = DeepQNetwork(n_features = QUANTITY_FEATURES, n_actions = QUANTITY_ACTIONS, 
			scope = 'camp_' + str(i) + '_global_ai', model = mlp, parent_ai = None, sess = Sess, 
			learning_rate = 5e-3, n_replace_target = 50, hiddens = HIDDENS, decay = 0.99, memory_size = 100000, batch_size = 2000, 
			epsilon_decrement = 1e-3, epsilon_lower = 0.001, learn_start = LEARN_START)

	# ============ local ai ======================
	for i in range(N_GAME):
		for j in range(2):
			local_ais[i].append(DeepQNetwork(n_features = QUANTITY_FEATURES, n_actions = QUANTITY_ACTIONS,
				scope = 'camp_' + str(j) + '_local_ai_' + str(i), model = mlp, parent_ai = global_ai[j], sess = Sess,
				hiddens = HIDDENS))

	# =========== restore or not ================
	Saver = tf.train.Saver()

	if RESTORE:
		Saver.restore(Sess, RESTORE_PATH)
		print('restored successfully from ' + RESTORE_PATH)
	else:
		Sess.run(tf.global_variables_initializer())


	# ========= multi thread start =============
	for i in range(N_GAME):		
		new_thread = td.Thread(target = play, args = (IP, PORT[i], local_ais[i], global_ai, dataQ))
		new_thread.start()


	# ======== main thread in learning =========
	while LEARNING:
		counter += 1
		for _ in range(TRAIN_TIME_EACH):
			fetch_data(global_ai[THIS_TURN], dataQ[THIS_TURN])
			if not ONLY_PLAY:
				global_ai[THIS_TURN].learn()
	
		save_path = Saver.save(Sess, SAVE_PATH + str(counter) + '.ckpt')
		print('saved in' + save_path)

		THIS_TURN = 1 - THIS_TURN