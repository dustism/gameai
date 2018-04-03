import sys
sys.path.append('../')

from LOLEnv import LOLEnv
from utility import *
from config import *

import tensorflow as tf
import threading as td
import queue

class SuperEnv():
	def __init__(self, n_games, players, sess):
		# ======== initialization =======
		self.n_games = n_games
		self.players_global = players
		self.sess = sess
		
		self.n_players = len(players)
		self.learn_steps = [0 for i in range(self.n_players)]
		self.learn_switches = [False for i in range(self.n_players)]

		self.players_local = [[self.players_global[i].act_only_copy() for i in range(self.n_players)] for _ in range(n_games)]

		self.dataQ = [queue.Queue() for _ in range(self.n_players)]
		self.scoreQ = queue.Queue()
		# ====== save and restore ======
		self.Saver = tf.train.Saver(max_to_keep = None)
		if RESTORE:
			self.Saver.restore(sess, RESTORE_PATH)
			print('restore successfully from ' + RESTORE_PATH)
		else:
			sess.run(tf.global_variables_initializer())


	def start_game(self):
		# ===== playing threads ======
		for i in range(self.n_games):
			new_thread = td.Thread(target = self._play, 
				args = (IP, PORT[i], self.players_local[i], 
					self.players_global, self.dataQ, self.scoreQ))
			new_thread.start()
		
		
		# ===== learning threads =====
		for i in range(self.n_players):
			new_thread = td.Thread(target = self._learn, args = (i, ))
			new_thread.start()

		
		td.Thread(target = self._plot, args = (self.scoreQ, )).start()


	def _plot(self, scoreQ):
		plotter = ScorePlotter()
		while True:
			score = scoreQ.get()
			plotter.plot(score)


	# ===== learn or not ======
	def set_learning(self, player_id, learn_switch):
		self.learn_switches[player_id] = learn_switch


	def _play(self, ip, port, local_ai, global_ai, dataQ, scoreQ):
		# ========= initializer ========================
		env = LOLEnv(ip, port)
		episode = 0
		history = (History(HISTORY_LENGTH), History(HISTORY_LENGTH))
		
		while NEW_EPISODE:
			# ===== re-initialize =======================
			episode += 1
			frame_counter = 0
			time_start = time.time()

			# ===== get global parameters ===============
			for i in range(self.n_players):
				history[i].clear()
				synchronize_version(local_ai[i], global_ai[i])

			_, observation = env.reset()
			feature = extract_feature(observation, frame_counter % self.n_players)
			
			#assert get_camp(observation.heroInfo[0].refreshID) == CAMP_RED
			
			while NEXT_FRAME:
				current_player = frame_counter % self.n_players
				next_player = (frame_counter + 1)% self.n_players
			
				# =========== ai make decision ==========
				action = local_ai[current_player].act(feature)
				action_id, action_wrapped = wrap_action(action, observation, current_player)


				# =========== get next feature ==========
				observation_next = consume_frame(SKIP_FRAME, env, action_id, action_wrapped)
				winner = judge_winner(observation_next)
				feature_next_next_p = extract_feature(observation_next, next_player)
				
				if self.learn_switches[current_player]:
					feature_next_current_p = extract_feature(observation_next, current_player)
					# =========== winner reward and done ====
					reward = shape_reward(observation, observation_next, winner, current_player)
					done = 1. if winner is not None else 0.
					# ==== same one training and acting =====
					if action_wrapped is not None:
						history[current_player].put((feature, action, reward, feature_next_current_p, done))
						if history[current_player].full() : dataQ[current_player].put(history[current_player].get())

				# === is game end or not ================
				if winner is not None or frame_counter >= MAX_EPISODES:
					# ====== time, frames, scores and epsilon ======
					time_cost = time.time() - time_start
					
					score = make_score(observation_next)
					scoreQ.put(score)

					print('Episode {} end, time : {}, every player has {} frames in 1 s'.format(episode, time_cost, frame_counter / self.n_players / time_cost))
					print('Score : {}'.format(score))
					print('epsilon(roughly): {}'.format((local_ai[0].epsilon + local_ai[1].epsilon) / 2))
					
					env.end()
					break

				observation = observation_next
				feature = feature_next_next_p
				frame_counter += 1
		
	def _learn(self, player_id):
		while LEARNING:
			fetch_data(self.players_global[player_id], self.dataQ[player_id])
			if not ONLY_PLAY:
				self.players_global[player_id].learn()
				self.learn_steps[player_id] += 1
				if self.learn_steps[player_id] % SAVE_EVERY == 0:
					save_path = self.Saver.save(self.sess, SAVE_PATH + str(sum(self.learn_steps)) + '.ckpt')
					print('saved in ' + save_path)