from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import copy

from ..constants.config import *


class FrameMonitor:
    """ Monitor frames per second for each player that averaged over a whole episode. """
    def __init__(self):
        self.times = 0
        self.init_frames = 0
        self.recent_frames = 0

    def few_frames(self, frames):
        """
        Record frames per second for each player in the newly end episode,
        and judge whether frames decrease a lot from the beginning.
        """
        # record from the most recent 10 episodes
        avg_over = 10
        if self.times < avg_over:
            self.times += 1
            self.init_frames += frames / avg_over
            self.recent_frames += frames / avg_over
        else:
            # update according to "(1-a)*old+a*new", here a=0.2
            self.recent_frames = self.recent_frames * 0.8 + frames * 0.2

        # threshold: 0.8
        return True if self.recent_frames < self.init_frames * 0.8 else False


class SARS:
    """ Reward shaping, trajectory saving... All the trivial matters are done within this class. """
    def __init__(self, history, data_queue, camp):
        self.history = history
        self.data_queue = data_queue
        self.camp = camp

        self.obs0 = None

        self.s0 = None
        self.a = None
        self.r = 0
        self.s1 = None

    def get_action(self, action):
        self.a = action

    def end(self, obs):
        self.s1 = obs.extract_feature(self.camp, SOLDIERS_CONSIDER)
        winner = obs.judge_winner()

        # when feature is formed first time, namely there's no s0, just skip
        if self.s0 is not None:
            done = 1. if winner is not None else 0.
            self.r = obs.shape_reward(self.obs0, self.camp, winner)
            self.history.put((self.s0, self.a, self.r, self.s1, done))
            if self.history.full():
                self.data_queue.put(self.history.get())

        # renew the start state and observation
        self.s0 = self.s1
        self.obs0 = copy.deepcopy(obs)

        return self.r


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object)
        self.index = -1

    def store(self, experience):
        self.index += 1
        self.data[self.index % self.capacity] = experience

    def sample(self, batch_size):
        sample_index = np.random.choice(min(self.index, self.capacity), size=batch_size)
        return self.data[sample_index]


class History:
    def __init__(self, length):
        self.history = np.zeros(length, dtype=object)
        self.length = length
        self.index = 0
        self.total = 0

    def put(self, data):
        self.history[self.index] = data
        self.index = (self.index + 1) % self.length
        self.total += 1

    def get(self):
        if not self.full():
            print('Warning : fetch data from uncompleted history')
            input('Press any button to continue')
        return np.append(self.history[self.index:], self.history[: self.index])

    def full(self):
        return self.total >= self.length

    def clear(self):
        self.total = 0
        self.index = 0


class ScorePlotter:
    def __init__(self):
        self.recent_twenty = []
        self.score_history = []
        plt.ion()
        self.fig = plt.figure()
        self.x_upperlim = 20
        self.ax = plt.axes(xlim=(0, self.x_upperlim), ylim=(-20, 15))
        self.ax.grid()
        self.line, = self.ax.plot([], [], lw=2)
        plt.pause(0.02)

    @property
    def plotter_animate(self):
        self.line.set_data(range(len(self.score_history)), self.score_history)
        if len(self.score_history) > self.x_upperlim:
            self.x_upperlim += 20
            self.ax.set_xlim(0, self.x_upperlim)
            self.ax.figure.canvas.draw()
        return self.line,

    def plot(self, score):
        self.recent_twenty.append(score)
        if len(self.recent_twenty) >= 50:
            self.score_history.append(np.mean(self.recent_twenty))
            animation.FuncAnimation(self.fig, self.plotter_animate, blit=False)
            del self.recent_twenty[0]
            plt.pause(0.02)


def plot_score(score_plotter, score_queue):
    if not score_queue.empty():
        while not score_queue.empty():
            score = score_queue.get()
            score_plotter.plot(score)


def synchronize_version(local_ai, global_ai):
    if local_ai.learn_step != global_ai.learn_step:
        local_ai.sync()


def fetch_data(ai, data_queue):
    """Every 100 frames train one time."""
    for i in range(100):
        exp = data_queue.get()
        ai.store(exp)
