from ..utils.game_utils import *
from ..constants.config import *
from ..constants.Defines import *


class LOLAI:
    def __init__(self, camp):
        self.camp = camp
        self.attack_range = ATTACK_RANGE_HERO

    def act(self, obs, hero_i):
        self_hero = obs.heroes[self.camp][hero_i]
        rd = round(random.uniform(0, 1))
        if obs.heroes[1 - self.camp][rd].alive:
            enemy_hero = obs.heroes[1 - self.camp][rd]  # randomly choose one from two enemies if both alive
        else:
            enemy_hero = obs.heroes[1 - self.camp][1 - rd]
        self_tower = obs.towers[self.camp]
        enemy_tower = obs.towers[1 - self.camp]
        if not self_hero.movable or not self_hero.alive:
            return Action.idle()

        self_soldiers = obs.get_soldiers_by_camp(self.camp)
        safe = False
        for self_soldier in self_soldiers:
            x_diff = self_soldier.place.x - self_hero.place.x
            z_diff = self_soldier.place.z - self_hero.place.z
            if self.camp == CAMP_BLUE and x_diff > 1 and z_diff > 1:
                safe = True
            elif self.camp == CAMP_RED and x_diff < -1 and z_diff < -1:
                safe = True

        if not safe:
            return Action.move_to(self_hero, self_tower.place)

        hero_dist = Observation.dis(self_hero.place, enemy_hero.place)
        if hero_dist < self.attack_range and random.uniform(0, 1) < 0.5:
            return Action.attack(self_hero, enemy_hero)

        tower_dist = Observation.dis(self_hero.place, enemy_tower.place)
        if tower_dist < self.attack_range:
            return Action.attack(self_hero, enemy_tower)

        sorted_soldiers = obs.get_soldiers(self_hero.place)
        if len(sorted_soldiers[1 - self.camp]) > 0:
            nearest_soldier_dist = Observation.dis(self_hero.place, sorted_soldiers[1 - self.camp][0].place)
            if nearest_soldier_dist < self.attack_range:
                return Action.attack(self_hero, sorted_soldiers[1 - self.camp][0])

        return Action.move_to(self_hero, enemy_tower.place)
