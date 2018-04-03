import AIEnv
import whole_pb2
from Defines import *

class LOLEnv:
	def __init__(self, ip, port):
		self.AIEnv = AIEnv.AIEnv(ip, port)

	def reset(self):
		self.AIEnv.recv()
		self.AIEnv.send(S2C_GameStartRequest_ID, None)
		_, game_start_info = self.AIEnv.recv()
		self.AIEnv.send(S2C_IDLE_ID, None)
		_, frame = self.AIEnv.recv()
		return game_start_info, frame

	def step(self, action_id, action):
		self.AIEnv.send(action_id, action)
		message_id, message = self.AIEnv.recv()
		return message	
		

	def end(self):
		self.AIEnv.send(S2C_GameEnd_ID, None)
