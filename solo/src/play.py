from LOLEnv import LOLEnv
from LOLAI import LOLAI

env = LOLEnv('172.16.9.136', 12345)
ai  = LOLAI()
counter = 0

while True:
	try:
		gameStartInfo, observation = env.reset()
		while True:
			counter = counter + 1
			#print(observation)
			if not observation.heroInfo[0].alive or counter == 1000:
				env.end()
				break
			messageID, message = ai.make_decision(observation)
			observation = env.step(messageID, message)			
	except:
		env = LOLEnv()
