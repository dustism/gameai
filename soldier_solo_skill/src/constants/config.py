# ---adjusting parameters
import time
import collections


Place = collections.namedtuple('Place', 'x z')

# ====== About the game =======
ATTACK_RANGE_HERO = [6., 6.]
SKILL_RANGE_HERO = [10., 10.]
TOWER_ID = [40132, 40232]
ATTACK_RANGE_TOWER = 8
PLACE_TOWER = [Place(x=-12.2, z=-8.6), Place(x=12.2, z=8.6)]

# ====== About the network =========
HIDDENS = [64, 128, 256, 128]
LEARNING_RATE = 5e-3
QUANTITY_FEATURES = 76
QUANTITY_ACTIONS = 16

# ====== About communications ======

SKIP_FRAME = 2
N_GAMES = 20
IP = '59.78.31.93'
PORT = [12345 + i for i in range(N_GAMES)]


# ===== About the scene ===========
NORD = 1.0
SOLDIERS_CONSIDER = 5
PLAYERS_EVERY_SIDE = 1
MAX_FRAMES_PER_EPISODE = 3000
HISTORY_LENGTH = 10


# ==== About training process ====
HISTORY_SAMPLE_BOUND = 0.8


# ====== Save and restore ========

SAVE_EVERY = 2000
SAVE_PATH = 'saved_models/' + time.ctime() + '/'

RESTORE = True
RESET_EPSILON = False
ONLY_PLAY = True
RESTORE_PATH = 'saved_models/Sat May 19 12:03:09 2018/6001.ckpt'
