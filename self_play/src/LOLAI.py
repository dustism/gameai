from Defines import *
import whole_pb2
import random

class LOLAI:
	def __init__(self):
		pass
		
	def randomMove(self, observation):
		messageID = S2C_HeroMove_ID    
		message = whole_pb2.AI_Move()
		message.refreshID = 40121
		direction = message.direction
		direction.x = random.uniform(-1, 1) 
		direction.y = 0
		direction.z = random.uniform(-1, 1)
		direction.nord = 1.0
		return messageID, message
		
	def skill(self, button, observation):
		messageID = S2C_HeroDirectionSkill_ID
		message = whole_pb2.AI_DirectionSkill()
		message.refreshID = 40121
		message.button = button
		direction = message.direction
		direction.x = random.uniform(0, 1) + observation.heroInfo[0].place.x
		direction.y = 0
		direction.z = random.uniform(0, 1) + observation.heroInfo[0].place.z
		direction.nord = random.randint(3, 5)
		return messageID, message
			
	def make_decision(self, observation):
		if (not observation.heroInfo[0].movable):
			#print('not able to move or skill')
			return S2C_IDLE_ID, None
		hero = observation.heroInfo[0]
		cd = {}
		skills = ['', 'W', 'E', 'R']
		
		for skill in hero.skillInfo:
			cd[skill.button] = skill.currentCD
		i = random.randint(0, 8)
		if i == 0:	
			return self.randomMove(observation)
		elif i <= 3:
			if cd[skills[i]] > 0:
				return S2C_IDLE_ID, None
			else:
				return self.skill(skills[i], observation)
		else:
			return S2C_IDLE_ID, None
			
	def attack(self, id):
		messageID = S2C_HeroTargetSkill_ID
		message = whole_pb2.AI_TargetSkill()
		message.refreshID = 40121
		message.button = 'A'
		message.targetID = id
		return messageID, message
'''		
	def make_decision(self, observation):
		if not observation.heroInfo[0].movable:
			print('not able to move or skill')
			return S2C_IDLE_ID, None
		hero = observation.heroInfo[0]
		for skill in hero.skillInfo:
			if skill.button == 'A':
				if not skill.ready:
					return S2C_IDLE_ID, None
		soldierCounter = len(observation.soldierInfo)
		index = random.randint(0, soldierCounter - 1)
		id = observation.soldierInfo[index].identityID
		return self.attack(id)
'''		
