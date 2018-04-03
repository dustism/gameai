import collections
import sys
from matplotlib import pyplot as plt
from matplotlib import animation
sys.path.append('src')
from Defines import *
from config import *
import whole_pb2
import numpy as np
import collections
import math
import time

Place = collections.namedtuple('Place', 'x z')

class ScorePlotter():
	def __init__(self):
		self.score_history = []
		plt.ion()
		self.fig = plt.figure()
		self.x_upperlim = 20
		self.ax = plt.axes(xlim=(0, self.x_upperlim), ylim=(-1.1, 1.1))
		self.ax.grid()
		self.line, = self.ax.plot([], [], lw=2)
		plt.pause(0.02)
		print('initialized')

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

def consume_frame(skip_frame, env, action_id, action):
	observation = env.step(action_id, action)
	for i in range(skip_frame):
		observation = env.step(S2C_IDLE_ID, None)
		if judge_winner(observation) is not None:
			break
			
	return observation

def synchronize_version(local_ai, global_ai):
	if local_ai.learn_step != global_ai.learn_step : 
		local_ai.sync()

def fetch_data(ai, dataQ, player_id = None):
	for _ in range(200):
		exp = dataQ.get()
		ai.store(exp)

	if player_id is not None:
		print('player {} sampled exp {}'.format(player_id, exp))
#	if dataQ.qsize() > 10000:
#		print('warning : data queue is getting bigger and bigger')

	#print('After fetching, the queue left {} experience'.format(dataQ.qsize()))

def shape_reward(observation, observation_next, winner, player_id):

	Encourage_Attack = 0.001


	# 1 for blue and 2 for red
	camp = get_camp(observation.heroInfo[player_id].refreshID)
	'''
	if winner == camp:
		return 2
	elif winner == 3 - camp:
		return -2
	'''

	# encourage attack
	coef = [0, 1, -1]
	hero_place_reward  = Encourage_Attack * place_difference_sum(observation.heroInfo[player_id].place, observation_next.heroInfo[player_id].place) * coef[camp]

	# health difference
	hero_health_reward = (health_difference(observation_next, camp) - health_difference(observation, camp)) / 1000. 

	return hero_place_reward + hero_health_reward


def judge_winner(observation):
	
	for hero in observation.heroInfo:
		# camp_blue = 1, camp_red = 2
		camp = get_camp(hero.refreshID)
		if not hero.alive : return 3 - camp

	return None
	
def place_difference_sum(place_1, place_2):
	return place_2.x - place_1.x + place_2.z - place_1.z

ATTACK_RANGE = [6., 6.]
ATTACK_RANGE_TOWER = 8.
PLACE_TOWER = [0, Place(x = -12.2, z = -8.6), Place(x = 12.2, z = 8.6)]
	
def extract_feature(observation, player_id):
	feature = [ATTACK_RANGE[player_id], ATTACK_RANGE[1 - player_id], ATTACK_RANGE_TOWER]

	camp = get_camp(observation.heroInfo[player_id].refreshID)
	
	place_self = observation.heroInfo[player_id].place
	place_enemy = observation.heroInfo[1 - player_id].place
	place_self_tower_1 = PLACE_TOWER[camp]
	place_enemy_tower_1 = PLACE_TOWER[3 - camp]

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
	
	feature.append(observation.heroInfo[player_id].heroAttribute.currentHealth / 1000.)
	feature.append(observation.heroInfo[1 - player_id].heroAttribute.currentHealth / 1000.)

	return np.array(feature)


def wrap_action(action, observation, player_id):
	assert 0 <= action < QUANTITY_ACTIONS

	if not observation.heroInfo[player_id].movable:
		return S2C_IDLE_ID, None

	if action == 0:
		dist = dis(observation.heroInfo[player_id].place, observation.heroInfo[1 - player_id].place)
		if dist > ATTACK_RANGE[player_id]:
			return move_to(observation.heroInfo[player_id], dir(observation.heroInfo[player_id].place, observation.heroInfo[1 - player_id].place))
		else:
			return attack(observation.heroInfo[player_id], observation.heroInfo[1 - player_id])
	else:
		angle = 2 * math.pi  / (QUANTITY_ACTIONS - 1) * action 
		return move_to(observation.heroInfo[player_id], Place(x = math.cos(angle), z = math.sin(angle)))	

def make_score(observation):
	return health_difference(observation, 2) / 1000.

def in_square(place, bound):
	return -bound < place.x < bound and -bound < place.z < bound

def in_circle(place, r):
	return dis(place, Place(x = 0., z = 0.)) < r

def get_camp(refreshID):
	return refreshID // 100 % 10 

def health_difference(observation, camp_standard):
	diff = 0
	# camp_blue = 1, camp_red = 2
	for hero in observation.heroInfo:
		camp = get_camp(hero.refreshID)
		if camp == camp_standard:
			mul = 1
		else:
			mul = -1
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
