from .whole_pb2 import *
from ..constants.Defines import *
from ..constants.config import *

import numpy as np
import collections
import random
import math

Place = collections.namedtuple('Place', 'x z')


class Unit(object):
    def __init__(self, refresh_id=0, health=0, place=None):
        """
        Abstract class for different kinds of units in the game, like hero, tower and soldier.

        :param refresh_id: for example, 40131 for tower, 40121 for hero, 40112 for soldier
        :param health: current HP for unit
        :param place: a 2-dim vector defined as `collections.namedtuple('Place', 'x z')`
        """
        self.refreshID = refresh_id
        self.health = health
        self.place = place


class Hero(Unit):
    def __init__(self, hero):
        """
        :param hero: store from `raw_obs.heroInfo[]`
        """
        super(Hero, self).__init__(hero.refreshID, hero.heroAttribute.currentHealth, hero.place)
        self.movable = hero.movable
        self.alive = hero.alive
        self.moveDirection = hero.moveDirection


class Tower(Unit):
    def __init__(self, tower, place):
        """
        :param tower: store from `raw_obs.towerInfo[]`
        :param place: place is not specified in raw_obs, so a `Place` vector need to be passed
        """
        super(Tower, self).__init__(tower.refreshID, tower.health, place)


class Soldier(Unit):
    def __init__(self, soldier):
        """
        :param soldier: store from `raw_obs.soldierInfo[]`
        """
        super(Soldier, self).__init__(soldier.refreshID, soldier.health, soldier.place)
        self.identityID = soldier.identityID


class Observation:
    def __init__(self):
        """
        Now designed for the solo case only.
        """
        self.heroes = [None, None]  # [40121 hero, 40221 hero]
        self.towers = [None, None]  # [40132 tower, 40232 tower], mid lane's first tower only
        self.soldiers = [{}, {}]  # the key is each soldier's identityID

    def reset(self):
        self.__init__()

    def build(self, raw_obs):
        """
        Refresh all the parameters from the new raw observation.

        :param raw_obs: the raw observation that game core returns.
        """
        self.reset()

        for hero_camp in range(2):
            self.heroes[hero_camp] = Hero(raw_obs.heroInfo[hero_camp])

        for tower in raw_obs.towerInfo:
            # Explicitly specify tower's place since there's no place info in raw_obs
            # For now only store the info of middle lane's first tower only
            if tower.refreshID == 40132:
                self.towers[0] = Tower(tower, PLACE_TOWER[0])
            elif tower.refreshID == 40232:
                self.towers[1] = Tower(tower, PLACE_TOWER[1])

        for soldier in raw_obs.soldierInfo:
            # Unlike hero and tower, soldiers differ from each other by their identityID, not refreshID
            self.soldiers[Observation.get_camp(soldier.refreshID)][soldier.identityID] = Soldier(soldier)

    def get_soldiers(self, hero_place):
        """
        Get all soldiers from both camps, soldiers of each camp are **SORTED** by distance from the selected hero.
        -- blue soldier refreshID: 40112, camp: 0
        -- red soldier refreshID: 40212, camp: 1

        :return: [[camp 0 soldiers], [camp 1 soldiers]]
        """
        soldiers = [list(self.soldiers[0].values()), list(self.soldiers[1].values())]

        for i in range(2):
            soldiers[i].sort(key=lambda x: Observation.dis(x.place, hero_place))

        return soldiers

    def get_soldiers_by_camp(self, camp):
        return list(self.soldiers[camp].values())

    def extract_feature(self, hero_camp, most_considered_soldiers=SOLDIERS_CONSIDER):
        """
        Do the feature extraction.

        :param most_considered_soldiers: specify how many nearest soldiers' features are considered
        :param hero_camp: specify which hero's perspective the feature is from.
        :return: a numpy array of features
        """
        feature = [ATTACK_RANGE_HERO[hero_camp], ATTACK_RANGE_HERO[1 - hero_camp], ATTACK_RANGE_TOWER]

        # =================== preparations ===================
        place_self = self.heroes[hero_camp].place
        place_enemy = self.heroes[1 - hero_camp].place
        place_self_tower = self.towers[hero_camp].place
        place_enemy_tower = self.towers[1 - hero_camp].place
        soldiers = self.get_soldiers(place_self)

        # ===================== features =====================
        Observation._append_vector(feature, place_self)

        Observation._dis_and_dir(feature, place_self, place_self_tower)
        Observation._dis_and_dir(feature, place_self, place_enemy_tower)
        Observation._dis_and_dir(feature, place_self, place_enemy)

        feature.append(self.heroes[hero_camp].health / 1000.)
        feature.append(self.heroes[1 - hero_camp].health / 1000.)
        feature.append(self.towers[hero_camp].health / 1000.)
        feature.append(self.towers[1 - hero_camp].health / 1000.)

        soldier_camp = hero_camp
        for _ in range(2):
            num_consider = min(len(soldiers[soldier_camp]), most_considered_soldiers)

            """
            Each soldier correspond to a 5-dim feature.

            1st:       valid bit, denoting if the soldier exists
            2nd:       soldier's distance from the hero
            3rd & 4th: soldier's direction from the hero
            5th:       soldier's current health
            """
            for i in range(num_consider):
                feature.append(1)
                Observation._dis_and_dir(feature, place_self, soldiers[soldier_camp][i].place)
                feature.append(soldiers[soldier_camp][i].health / 1000.)

            # fill the non-existing nearest soldiers' feature with all-zero vector
            if len(soldiers[soldier_camp]) < most_considered_soldiers:
                for _ in range(most_considered_soldiers - len(soldiers[soldier_camp])):
                    feature = feature + [0, 0, 0, 0, 0]

            # now switch to consider the enemy soldiers' features
            soldier_camp = 1 - soldier_camp

        return np.array(feature)

    def judge_winner(self, tower_least_hp=3850):
        """
        First judge whether any camp's hero is dead, then judge whether any camp's tower is broken down.

        :param tower_least_hp: a tower is defined as broken when its health is below this parameter.
        :return: winner's camp, or None if no winner
        """
        for hero_camp in range(2):
            if not self.heroes[hero_camp].alive:
                return 1 - hero_camp  # since camp can only be 0 or 1

        for tower_camp in range(2):
            if self.towers[tower_camp].health < tower_least_hp:
                return 1 - tower_camp

        return None

    def shape_reward(self, obs_prev, camp, winner):
        """
        Reward shaping from two continuous observations.

        :param obs_prev: previous observation
        :param camp: 0(blue) or 1(red), specify which hero's perspective the reward is from
        :param winner: return an additional end reward if winner is not None
        """

        winner_score = 0
        if winner is not None:
            if winner == camp:
                winner_score = 10
            elif winner == 1 - camp:
                winner_score = -5

        # ========================  tower_score ========================
        tower_score = - (obs_prev.towers[camp].health - obs_prev.towers[1 - camp].health) / 100.
        tower_score += (self.towers[camp].health - self.towers[1 - camp].health) / 100.

        # ======================== health_score ========================
        health_score = (self.hero_hp_diff(camp, for_reward=False) - obs_prev.hero_hp_diff(camp, for_reward=False)) / 1000.

        # ====================== encourage_attack ======================
        encourage_attack = 0.
        if self.hero_place_diff() > 10.:
            encourage_attack = - math.log(self.hero_place_diff() / 10) / 500.

        return winner_score + tower_score + health_score + encourage_attack - 0.004

    def hero_hp_diff(self, camp, for_reward=True):
        """
        Compute health difference between two heroes.

        :param camp: 0(blue) or 1(red), specify which hero's perspective the difference is from.
        :param for_reward: in order to encourage attacking, enemy's HP reduction will be magnified.
        """
        diff = 0
        for hero in self.heroes:
            hero_camp = Observation.get_camp(hero.refreshID)
            if camp == hero_camp:
                mul = 1
            else:
                mul = -1 if not for_reward else -3
            diff += mul * hero.health

        return diff

    def hero_place_diff(self):
        return Observation.dis(self.heroes[0].place, self.heroes[1].place)

    @staticmethod
    def dis(place_1, place_2):
        """Compute distance of two place vector."""
        return ((place_1.x - place_2.x) ** 2 + (place_1.z - place_2.z) ** 2) ** 0.5

    @staticmethod
    def dir(place_1, place_2):
        """Compute direction from place_1 pointing to place_2, the return is a norm=1 <Place> vector."""
        dist = Observation.dis(place_1, place_2)
        if dist == 0:
            # Note that sometimes two vector may perfectly coincide.
            return Place(x=0, z=0)
        return Place(x=(place_2.x - place_1.x) / dist, z=(place_2.z - place_1.z) / dist)

    @staticmethod
    def discretize(direction, n):
        """
        Discretize a vector, for example, if n=8, then vector with angle -22.5~22.5 will be classified to 8,
        vector with angle 22.5~67.5 will be classified to 1.

        :param direction: <Place> type vector
        :param n: num of discrete values
        """
        d = Observation.dis(Place(x=0, z=0), direction)
        assert d != 0

        if direction.z > 0:
            angle = math.acos(direction.x / d)
        else:
            angle = 2 * math.pi - math.acos(direction.x / d)

        discrete_value = round(angle / (2*math.pi/n))
        if discrete_value == 0:
            discrete_value = n

        return discrete_value


    @staticmethod
    def _dis_and_dir(feature, vector_1, vector_2):
        """Auxiliary method for feature extraction only"""
        direction = Observation.dir(vector_1, vector_2)
        Observation._append_vector(feature, direction)
        feature.append(Observation.dis(vector_1, vector_2))

    @staticmethod
    def _append_vector(feature, vector):
        """Auxiliary method for feature extraction only"""
        feature.append(vector.x)
        feature.append(vector.z)

    @staticmethod
    def in_attack_range(place_1, place_2, attack_range):
        return Observation.dis(place_1, place_2) <= attack_range

    @staticmethod
    def in_square(place, bound):
        return -bound < place.x < bound and -bound < place.z < bound

    @staticmethod
    def in_circle(place, r):
        return Observation.dis(place, Place(x=0., z=0.)) < r

    @staticmethod
    def get_camp(refresh_id):
        return refresh_id // 100 % 10 - 1

    def consume_frame(self, skip_frame, env, action):
        """
        Skip some frames to reduce similar observations.

        :param skip_frame: num_frame that will be skipped
        :param env: LOLEnv
        :param action: a tuple defined in game, <messageID, message>
        """
        message_id, message = action

        self.build(env.step(message_id, message))
        for _ in range(skip_frame):
            if self.judge_winner() is not None:
                break  # break if the game ends according to our definition
            self.build(env.step(message_id, message))


class Action:
    # ============ three basic actions ============
    @staticmethod
    def idle():
        """Do nothing."""
        return S2C_IDLE_ID, None

    @staticmethod
    def move(hero, direction):
        """
        Make a hero move according to a direction.

        :param hero: class <Hero>
        :param direction: movement direction
        """
        move = AI_Move()
        move.refreshID = hero.refreshID
        move.direction.x = direction.x
        move.direction.y = 0
        move.direction.z = direction.z
        move.direction.nord = 1.0

        return S2C_HeroMove_ID, move

    @staticmethod
    def attack(hero, target):
        """
        Make a hero attack the target.

        :param hero: class <Hero>
        :param target: any subclass of <Unit>
        """
        message = AI_TargetSkill()
        message.refreshID = hero.refreshID
        message.button = 'A'
        if hasattr(target, 'identityID'):
            message.targetID = target.identityID
        else:
            message.targetID = target.refreshID
        return S2C_HeroTargetSkill_ID, message

    @staticmethod
    def skill(hero, target, button):
        """
        Make a hero use a certain skill on the target.

        :param hero: class <Hero>
        :param target: any subclass of <unit>
        :param button: "Q", "W", "E", "R"
        """
        message = AI_DirectionSkill()
        message.refreshID = hero.refreshID
        normdir = Observation.dir(hero.place, target.place)
        message.button = 'W'
        #message.direction.x = normdir.x * hero.moveDirection.x - normdir.z * hero.moveDirection.z
        #message.direction.z = normdir.z * hero.moveDirection.x + normdir.x * hero.moveDirection.z
        message.direction.x = normdir.x
        message.direction.z = normdir.z
        message.direction.y = 0
        #print()
        #print(normdir)
        #print(message.direction.x, message.direction.z)
        return S2C_HeroDirectionSkill_ID, message


    # ============ auxiliary actions ============
    @staticmethod
    def random_walk(hero):
        angle = random.uniform(-math.pi, math.pi)
        return Action.move(hero, Place(x=math.cos(angle), z=math.sin(angle)))

    @staticmethod
    def walk_to_center(hero):
        return Action.move(hero, Place(0, 0))

    @staticmethod
    def want_to_attack(hero, target, attack_range):
        """First move closer to the target straightly, then attack."""
        dist = Observation.dis(hero.place, target.place)
        if dist < attack_range:
            print(2221)
            return Action.attack(hero, target)
        else:
            print(2222)
            return Action.move(hero, Observation.dir(hero.place, target.place))

    @staticmethod
    def move_to(hero, place):
        """Move to a specific place."""
        return Action.move(hero, Observation.dir(hero.place, place))

    # ============ auxiliary functions ============
    @staticmethod
    def wrap_action(action, obs, camp):
        """Used to convert NN outputs to legal actions in game."""
        assert 0 <= action < QUANTITY_ACTIONS

        if not obs.heroes[camp].movable:
            return Action.idle()

        self_hero = obs.heroes[camp]
        enemy_hero = obs.heroes[1 - camp]

        if action == 0:
            # action 0 is defined to be ATTACK
            hero_dist = Observation.dis(self_hero.place, enemy_hero.place)
            if hero_dist < ATTACK_RANGE_HERO[camp]:
                return Action.attack(self_hero, enemy_hero)
            # else:
            #     return Action.move(self_hero, Observation.dir(self_hero.place, enemy_hero.place))
            # """Note that for this version the controlled hero attack enemy hero only."""

            tower_dist = Observation.dis(self_hero.place, PLACE_TOWER[1 - camp])
            if tower_dist < ATTACK_RANGE_HERO[camp]:
                return Action.attack(self_hero, obs.towers[1 - camp])

            soldiers = obs.get_soldiers(self_hero.place)
            if len(soldiers[1 - camp]) > 0:
                nearest_soldier_dist = Observation.dis(self_hero.place, soldiers[1 - camp][0].place)
                if nearest_soldier_dist < ATTACK_RANGE_HERO[camp]:
                    return Action.attack(self_hero, soldiers[1 - camp][0])

            return Action.move(self_hero, Observation.dir(self_hero.place, enemy_hero.place))
        elif action == 1:
            # action 1 is defined to be use skill
            hero_dist = Observation.dis(self_hero.place, enemy_hero.place)
            if hero_dist < ATTACK_RANGE_HERO[camp]:
                return Action.skill(self_hero, enemy_hero, 'W')

            soldiers = obs.get_soldiers(self_hero.place)
            if len(soldiers[1 - camp]) > 0:
                nearest_soldier_dist = Observation.dis(self_hero.place, soldiers[1 - camp][0].place)
                if nearest_soldier_dist < ATTACK_RANGE_HERO[camp]:
                    return Action.skill(self_hero, soldiers[1 - camp][0], 'W')

            return Action.move(self_hero, Observation.dir(self_hero.place, enemy_hero.place))
        else:
            angle = 2 * math.pi / (QUANTITY_ACTIONS - 1) * (action - 1)
            return Action.move(self_hero, Place(x=math.cos(angle), z=math.sin(angle)))
