# ---adjusting parameters
import time
import collections


Place = collections.namedtuple('Place', 'x z')

# ====== About the game =======
ATTACK_RANGE_HERO = [6, 6.]
TOWER_ID = [40132, 40232]
ATTACK_RANGE_TOWER = 8
PLACE_TOWER = [Place(x=-12.2, z=-8.6), Place(x=12.2, z=8.6)]

# ====== About the network =========
HIDDENS = [64, 128, 256, 128]
LEARNING_RATE = 5e-3
QUANTITY_FEATURES = 68
QUANTITY_ACTIONS = 9

# ====== About communications ======

SKIP_FRAME = 2
N_GAMES = 50
IP = '172.16.8.111'
PORT = [12345 + i for i in range(N_GAMES)]


# ===== About the scene ===========
NORD = 1.0
SOLDIERS_CONSIDER = 5
PLAYERS_EVERY_SIDE = 2
MAX_FRAMES_PER_EPISODE = 3000
HISTORY_LENGTH = 10


# ==== About training process ====
HISTORY_SAMPLE_BOUND = 0.8


# ====== Save and restore ========

SAVE_EVERY = 500
SAVE_PATH = 'saved_models/' + time.ctime() + '/'

RESTORE = True
RESET_EPSILON = True
ONLY_PLAY = False
RESTORE_PATH = 'saved_models/beat_defensive_ai_2ed/2001.ckpt'
LEARN_START = 0
