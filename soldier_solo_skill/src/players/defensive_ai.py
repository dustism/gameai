from ..utils.game_utils import *
from ..constants.config import *
from ..constants.Defines import *


class LOLAI:
    def __init__(self, camp):
        self.camp = camp

    def act(self, obs):
        self_hero = obs.heroes[self.camp]
        enemy_hero = obs.heroes[1 - self.camp]
        self_tower = obs.towers[self.camp]
        enemy_tower = obs.towers[1 - self.camp]

        # idle if cannot move
        if not self_hero.movable:
            return Action.idle()

        self_soldiers = obs.get_soldiers_by_camp(self.camp)
        safe = False
        # make sure the hero always hide behind the all the soldiers at least 1 unit distance
        for self_soldier in self_soldiers:
            x_diff = self_soldier.place.x - self_hero.place.x
            z_diff = self_soldier.place.z - self_hero.place.z
            if self.camp == CAMP_BLUE and x_diff > 1 and z_diff > 1:
                safe = True
            elif self.camp == CAMP_RED and x_diff < -1 and z_diff < -1:
                safe = True
            # always hide unless the hero has been very close to tower
            elif Observation.in_circle(self_hero.place, ATTACK_RANGE_TOWER - ATTACK_RANGE_HERO[1-self.camp],
                                       self_tower.place):
                safe = True

        # not safe if can be attacked by enemy tower also
        if Observation.in_circle(self_hero.place, ATTACK_RANGE_TOWER, enemy_tower.place):
            safe = False

        if not safe:
            return Action.move_to(self_hero, self_tower.place)

        hero_dist = Observation.dis(self_hero.place, enemy_hero.place)
        # with some probability attack enemy hero, since tower and soldier should also be attacked
        if random.uniform(0, 1) < 1.0:
            if hero_dist < SKILL_RANGE_HERO[self.camp] and Action.skill_ready(self_hero, "W"):
                # use skill when available
                return Action.skill(self_hero, enemy_hero, 'W')
            elif hero_dist < ATTACK_RANGE_HERO[self.camp]:
                # otherwise attack
                return Action.attack(self_hero, enemy_hero)

        # attack tower if possible
        tower_dist = Observation.dis(self_hero.place, enemy_tower.place)
        if tower_dist < ATTACK_RANGE_HERO[self.camp]:
            return Action.attack(self_hero, enemy_tower)

        sorted_soldiers = obs.get_soldiers_by_dis(self_hero.place)
        # attack only the nearest soldier within attack range, and never waste skill on soldiers
        if len(sorted_soldiers[1 - self.camp]) > 0:
            nearest_soldier_dist = Observation.dis(self_hero.place, sorted_soldiers[1 - self.camp][0].place)
            if nearest_soldier_dist < ATTACK_RANGE_HERO[self.camp]:
                return Action.attack(self_hero, sorted_soldiers[1 - self.camp][0])

        return Action.move_to(self_hero, enemy_tower.place)
