from ..utils.game_utils import *
import random


class LOLAI:
    def __init__(self, camp):
        self.camp = camp

    def act(self, obs):
        self_hero = obs.heroes[self.camp]
        if not self_hero.movable:
            return Action.idle()

        """For probability 0.8 walk randomly, and the remained 0.2 walk to center."""
        if random.uniform(0, 1) > 0.2:
            return Action.random_walk(self_hero)
        else:
            return Action.walk_to_center(self_hero)

