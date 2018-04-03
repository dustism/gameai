import sys
if __name__ == '__main__':
	sys.path.append('src')
	sys.path.append('models')
from SuperEnv import SuperEnv
from learn import DeepQNetwork
from utility import *
from Defines import *
from config import *
from models import *

import tensorflow as tf

if __name__ == '__main__':
	Sess = tf.Session()

	players = (
		DeepQNetwork(n_features = QUANTITY_FEATURES, n_actions = QUANTITY_ACTIONS, 
				scope = 'camp_' + str(0) + '_global_0', model = mlp, 
				parent_ai = None, sess = Sess, learning_rate = 5e-3, 
				n_replace_target = 50, hiddens = HIDDENS, decay = 0.99, 
				memory_size = 100000, batch_size = 2000, epsilon_decrement = 1e-3,
				epsilon_lower = 0.001, learn_start = LEARN_START),
				
		DeepQNetwork(n_features = QUANTITY_FEATURES, n_actions = QUANTITY_ACTIONS, 
				scope = 'camp_' + str(1) + '_global_0', model = mlp, 
				parent_ai = None, sess = Sess, learning_rate = 5e-3, 
				n_replace_target = 50, hiddens = HIDDENS, decay = 0.99, 
				memory_size = 100000, batch_size = 2000, epsilon_decrement = 1e-3,
				epsilon_lower = 0.001, learn_start = LEARN_START)
	)



if __name__ == '__main__':

	super_env = SuperEnv(n_games = N_GAMES, players = players, sess = Sess)
		
	super_env.start_game()

	
	# learning logic
	learning_player = 0
	super_env.set_learning(player_id = learning_player, learn_switch = True)
	learn_threshold = LEARNING_TURNS_EACH
	
	while LEARNING:
		time.sleep(1)
		if super_env.learn_steps[learning_player] > learn_threshold:
			for i in range(2):
				print(super_env.learn_steps[i])
			super_env.set_learning(player_id = learning_player, learn_switch = False)
			learning_player = 1 - learning_player
			super_env.set_learning(player_id = learning_player, learn_switch = True)
			learn_threshold = super_env.learn_steps[learning_player] + LEARNING_TURNS_EACH
			print('now player {} is in training'.format(learning_player))