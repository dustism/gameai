from src.env import AIEnv
from src.constants.Defines import *
import socket
import time


class LOLEnv:
    def __init__(self, ip, port):
        self.AIEnv = AIEnv.AIEnv(ip, port)
        self.send_time_count = 0.
        self.recv_time_count = 0.

    def reset(self):
        self.send_time_count = 0.
        self.recv_time_count = 0.
        self.AIEnv.recv()
        self.AIEnv.send(S2C_GameStartRequest_ID, None)
        _, game_start_info = self.AIEnv.recv()
        self.AIEnv.send(S2C_IDLE_ID, None)
        _, frame = self.AIEnv.recv()
        return game_start_info, frame

    def step(self, action_id, action):
        time_1 = time.time()
        self.AIEnv.send(action_id, action)
        time_2 = time.time()
        message_id, message = self.AIEnv.recv()
        time_3 = time.time()

        self.send_time_count += time_2 - time_1
        self.recv_time_count += time_3 - time_2
        return message

    def end(self):
        self.AIEnv.send(S2C_GameEnd_ID, None)

    def close(self):
        self.AIEnv.sock.shutdown(socket.SHUT_RDWR)
        self.AIEnv.sock.close()
