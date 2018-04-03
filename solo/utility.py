import collections
from Defines import *
from config import *
import whole_pb2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import collections
import math
import time

Place = collections.namedtuple('Place', 'x z')
TOWER_X = 12.2
TOWER_Y = 8.6

class History():
	def __init__(self, length):
		self.history = np.zeros(length, dtype = object)
		self.length = length
		self.index = 0
		self.total = 0
		
	def put(self, data):
		self.history[self.index] = data
		self.index = (self.index + 1) % self.length
		self.total += 1
		
	def get(self):
		if not self.full():
			print('warning : fetch data from uncomplete history')
			input('Press any button to continue')
		return np.append(self.history[self.index : ], self.history[ : self.index])

	def full(self):
		return self.total >= self.length

	def clear(self):
		self.total = 0
		self.index = 0


class ScorePlotter():
	def __init__(self):
		self.score_history = []
		plt.ion()
		self.fig = plt.figure()
		self.x_upperlim = 20
		self.ax = plt.axes(xlim=(0, self.x_upperlim), ylim=(-5, 5))
		self.ax.grid()
		self.line, = self.ax.plot([], [], lw=2)
		plt.pause(0.02)

	@property
	def ploter_animate(self):
		self.line.set_data(range(len(self.score_history)), self.score_history)
		if len(self.score_history) > self.x_upperlim:
			self.x_upperlim += 20
			self.ax.set_xlim(0, self.x_upperlim)
			self.ax.figure.canvas.draw()
		return self.line,

	def plot(self, score):
		self.score_history.append(score)
		animation.FuncAnimation(self.fig, self.ploter_animate, blit=False)
		plt.pause(0.02)


class QValuePlotter():
	def __init__(self):
		self.qvalue_history = []
		plt.ion()
		self.fig = plt.figure()
		self.x_upperlim = 20
		self.ax = plt.axes(xlim=(0, self.x_upperlim), ylim=(0, 5))
		self.ax.grid()
		self.line, = self.ax.plot([], [], lw=2)
		plt.pause(0.02)

	@property
	def ploter_animate(self):
		self.line.set_data(range(len(self.qvalue_history)), self.qvalue_history)
		if len(self.qvalue_history) > self.x_upperlim:
			self.x_upperlim += 20
			self.ax.set_xlim(0, self.x_upperlim)
			self.ax.figure.canvas.draw()
		return self.line,

	def plot(self, qvalue):
		self.qvalue_history.append(qvalue)
		animation.FuncAnimation(self.fig, self.ploter_animate, blit=False)
		plt.pause(0.02)

def evaluate_qvalue(global_ai, qvalue_plotter, evaluation_set):
	qvalue_sum = 0
	for feature in evaluation_set:
		qvalue_sum += global_ai.max_qvalue(feature)
	qvalue_mean = qvalue_sum / len(evaluation_set)
	print(qvalue_mean)
	qvalue_plotter.plot(qvalue_mean)

def consume_frame(skip_frame, env, action_id, action):
	observation = env.step(action_id, action)
	for i in range(skip_frame):
		observation = env.step(S2C_IDLE_ID, None)
		if judge_winner(observation) is not None:
			break
			
	return observation

def synchronize_version(local_ai, global_ai):
	if local_ai.learn_step != global_ai.learn_step : local_ai.sync()

def plot_score(score_plotter, scoreQ):
	while not scoreQ.empty():
		score = scoreQ.get()
		score_plotter.plot(score)

def fetch_data(ai, dataQ):
	for i in range(200):
		# print(i)
		exp = dataQ.get()
		ai.store(exp)
#	if dataQ.qsize() > 10000:
#		print('warning : data queue is getting bigger and bigger')

	#print('After fetching, the queue left {} experience'.format(dataQ.qsize()))

def shape_reward(observation, observation_next, winner, frame_counter):
	'''
	if winner == CAMP_RED:
		return 2
	elif winner == CAMP_BLUE:
		return -2
	'''
	fix = 0 if in_circle(observation.heroInfo[0].place, 15.) else -0.1

	return (health_difference(observation_next) - health_difference(observation)) / 1000. + fix


def judge_winner(observation):
	
	for hero in observation.heroInfo:
		# camp_blue = 1, camp_red = 2
		camp = get_camp(hero.refreshID)
		if not hero.alive : return 3 - camp

	return None

def extract_feature(observation):
	feature = [ATTACK_RANGE_SELF, ATTACK_RANGE_ENEMY, ATTACK_RANGE_TOWER]
	
	assert get_camp(observation.heroInfo[0].refreshID) == CAMP_RED

	place_self = observation.heroInfo[0].place
	place_enemy = observation.heroInfo[1].place
	place_self_tower_1 = Place(x = TOWER_X, z = TOWER_Y)
	place_enemy_tower_1 = Place(x = -TOWER_X, z = -TOWER_Y)

	append_vector(feature, place_self)
	'''
	append_vector(feature, place_enemy)
	append_vector(feature, place_self_tower_1)
	append_vector(feature, place_enemy_tower_1)
	'''
	
	dis_and_dir(feature, place_self, place_self_tower_1)
	dis_and_dir(feature, place_self, place_enemy_tower_1)
	dis_and_dir(feature, place_self, place_enemy)
	
	'''
	dis_and_dir(feature, place_enemy, place_self_tower_1)
	dis_and_dir(feature, place_enemy, place_enemy_tower_1)
	'''
	
	feature.append(observation.heroInfo[0].heroAttribute.currentHealth / 1000.)
	feature.append(observation.heroInfo[1].heroAttribute.currentHealth / 1000.)

	return np.array(feature)


def wrap_action(action, observation):
	assert 0 <= action < QUANTITY_ACTIONS

	if not observation.heroInfo[0].movable:
		return S2C_IDLE_ID, None

	if action == 0:
		dist = dis(observation.heroInfo[0].place, observation.heroInfo[1].place)
		if dist > ATTACK_RANGE_SELF:
			return move_to(observation.heroInfo[0], dir(observation.heroInfo[0].place, observation.heroInfo[1].place))
		else:
			return attack(observation.heroInfo[0], observation.heroInfo[1])
	else:
		angle = 2 * math.pi  / (QUANTITY_ACTIONS - 1) * action 
		return move_to(observation.heroInfo[0], Place(x = math.cos(angle), z = math.sin(angle)))	

def in_square(place, bound):
	return -bound < place.x < bound and -bound < place.z < bound

def in_circle(place, r):
	return dis(place, Place(x= 0., z = 0.)) < r

def self_make_decision(observation):
	dist = dis(observation.heroInfo[0].place, observation.heroInfo[1].place)
	if dist > ATTACK_RANGE_ENEMY * 2: return 0
	else:
		direction = discretize(dir(observation.heroInfo[0].place, Place(x = TOWER_X, z = TOWER_Y)))
		action = np.argmin(np.abs(direction - np.arange(0, 9)))
		if action == 0: action = 8
		return action
			
	
def enemy_make_decision(observation):
	dist = dis(observation.heroInfo[0].place, Place(x = TOWER_X, z = TOWER_Y))
	if dist < ATTACK_RANGE_TOWER or not in_circle(observation.heroInfo[0].place, 15.):
		return move_to(observation.heroInfo[1], dir(observation.heroInfo[1].place, Place(x = 0.0, z = 0.0)))

	dist = dis(observation.heroInfo[0].place, observation.heroInfo[1].place)
	if dist > ATTACK_RANGE_ENEMY:
		return move_to(observation.heroInfo[1], dir(observation.heroInfo[1].place, observation.heroInfo[0].place))
	else:
		return attack(observation.heroInfo[1], observation.heroInfo[0])

		

def get_camp(refreshID):
	return refreshID // 100 % 10


def health_difference(observation):
	
	diff = 0
	# camp_blue = 1, camp_red = 2
	for hero in observation.heroInfo:
		camp = get_camp(hero.refreshID)
		if camp == CAMP_RED:
			mul = 1
		else:
			mul = -3
		diff += mul * hero.heroAttribute.currentHealth

	return diff


def append_vector(feature, vector):
	feature.append(vector.x)
	feature.append(vector.z)


def dis(vector_1, vector_2):
	return ((vector_1.x - vector_2.x) ** 2 + (vector_1.z - vector_2.z) ** 2) ** 0.5


def dir(vector_1, vector_2):
	dist = dis(vector_1, vector_2)
	return Place(x = (vector_2.x - vector_1.x) / dist, z = (vector_2.z - vector_1.z) / dist)


def discretize(vector):
	if vector.z > 0:
		angle = math.acos(vector.x)
	else:
		angle = 2 * math.pi - math.acos(vector.x)

	return 4 * angle  / math.pi


def dis_and_dir(feature, vector_1, vector_2):
	direction = dir(vector_1, vector_2)
	append_vector(feature, direction)
	feature.append(dis(vector_1, vector_2))


def move_to(hero, vector):
	move = whole_pb2.AI_Move()
	move.refreshID = hero.refreshID
	move.direction.x = vector.x
	move.direction.y = 0
	move.direction.z = vector.z
	move.direction.nord = 1.0

	return S2C_HeroMove_ID, move


def attack(hero_1, hero_2):
	attack = whole_pb2.AI_TargetSkill()
	attack.refreshID = hero_1.refreshID
	attack.button = 'A'
	attack.targetID = hero_2.refreshID

	return S2C_HeroTargetSkill_ID, attack
