import gym
import tensorflow as tf
from learn import *
from models import *

env = gym.make('CartPole-v1')
ai = DeepQNetwork(n_features = 4, n_actions = 2, model = mlp, sess = tf.Session())
episode = 0
total_score = 0

while True:
	observation = env.reset()
	
	time = 0
	episode += 1	
	while True:
		# env.render()
		time += 1
		action = ai.act(observation)
		observation_next, reward, done, _ = env.step(action)
		if done and time < 500:
			reward = -20
		ai.store(observation, action, reward, observation_next, done)
		if done:
			ai.learn()
			total_score += time
			if episode % 100 == 0:
				print('in episode {}, ai score {}, epsilon {}'.format(episode, total_score / 100, ai.epsilon))
				total_score = 0
			break
		observation = observation_next