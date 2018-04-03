from ..utils.game_utils import *
from ..constants.config import *


class LOLAI:
    def __init__(self, camp):
        self.camp = camp
        self.attack_range = ATTACK_RANGE_HERO[camp]

    def act(self, obs):
        self_hero = obs.heroes[self.camp]
        if not self_hero.movable:
            return Action.idle()

        enemy_hero = obs.heroes[1 - self.camp]
        if Observation.in_attack_range(self_hero.place, enemy_hero.place, self.attack_range):
            return Action.attack(self_hero, enemy_hero)

        enemy_tower = obs.towers[1 - self.camp]
        if Observation.in_attack_range(self_hero.place, enemy_tower.place, self.attack_range):
            return Action.attack(self_hero, enemy_tower)

        enemy_soldiers = obs.get_soldiers(self_hero.place)[1 - self.camp]
        for enemy_soldier in enemy_soldiers:
            if Observation.in_attack_range(self_hero.place, enemy_soldier.place, self.attack_range):
                return Action.attack(self_hero, enemy_soldier)

        return Action.move(self_hero, Observation.dir(self_hero.place, enemy_hero.place))
