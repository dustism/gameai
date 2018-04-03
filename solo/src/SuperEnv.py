from utility import *
from config import *

import tensorflow as tf
import threading as td


class SuperEnv():
	def __init__(self, n_games, players, sess):
		# ======== initialization =======
		self.n_games = n_games
		self.players_global = players

		self.n_players = len(players)
		self.learn_steps = [0 for i in range(self.n_players)]
		self.learn_switches = [False for i in range(self.n_players)]

		self.players_local = [[self.players_global[i].act_only_copy() for i in range(self.n_players)] for _ in range(n_games)]

		# ====== save and restore ======
		Saver = tf.train.Saver()
		if RESTORE:
			Saver.restore(sess, RESTORE_PATH)
			print('restore successfully from ' + RESTORE_PATH)
		else:
			sess.run(tf.global_variables_initializer())


	def start_game(self):
		for i in range(self.n_games):
			new_thread = td.Thread(target = self._play, args = (IP, PORT[i], self.players))

	def set_learning(player_id, learn_switch):
		self.learn_switches[player_id] = learn_switch


	def _play(self):
		pass