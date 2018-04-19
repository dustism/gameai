from ..utils.game_utils import *
from ..constants.config import *
from ..constants.Defines import *


class LOLAI:
    def __init__(self, camp):
        self.camp = camp
        self.attack_range = ATTACK_RANGE_HERO[camp]

    def act(self, obs):
        self_hero = obs.heroes[self.camp]
        enemy_hero = obs.heroes[1 - self.camp]
        self_tower = obs.towers[self.camp]
        enemy_tower = obs.towers[1 - self.camp]
        if not self_hero.movable:
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
        if hero_dist < self.attack_range and random.uniform(0, 1) < 0.7:
            return Action.skill(self_hero, enemy_hero, 'W')
        elif hero_dist < self.attack_range and random.uniform(0, 1) < 0.3:
            return Action.skill(self_hero, enemy_hero, 'W')

        tower_dist = Observation.dis(self_hero.place, enemy_tower.place)
        if tower_dist < self.attack_range:
            return Action.attack(self_hero, enemy_tower)

        sorted_soldiers = obs.get_soldiers(self_hero.place)
        if len(sorted_soldiers[1 - self.camp]) > 0:
            nearest_soldier_dist = Observation.dis(self_hero.place, sorted_soldiers[1 - self.camp][0].place)
            if nearest_soldier_dist < self.attack_range and random.uniform(0, 1) < 0.7:
                return Action.attack(self_hero, sorted_soldiers[1 - self.camp][0])
            elif nearest_soldier_dist < self.attack_range:
                return Action.skill(self_hero, sorted_soldiers[1 - self.camp][0], 'W')
        #return Action.tskill(self_hero)
        return Action.move_to(self_hero, enemy_tower.place)
