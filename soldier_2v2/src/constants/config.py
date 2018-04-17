# ---adjusting parameters
import time
import collections


Place = collections.namedtuple('Place', 'x z')

# ====== About the game =======
ATTACK_RANGE_HERO = 6.
TOWER_ID = [40132, 40232]
ATTACK_RANGE_TOWER = 8.
PLACE_TOWER = [Place(x=-12.2, z=-8.6), Place(x=12.2, z=8.6)]

# ====== About the network =========
HIDDENS = [256, 128, 128, 64]
LEARNING_RATE = 5e-3
QUANTITY_FEATURES = 79
QUANTITY_ACTIONS = 16

# ====== About communications ======
SKIP_FRAME = 0
N_GAMES = 20
IP = '172.16.8.111'
PORT = [12346 + i for i in range(N_GAMES)]


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

RESTORE = False
RESET_EPSILON = False
ONLY_PLAY = False
RESTORE_PATH = 'saved_models/single_net/30001.ckpt'
LEARN_START = 0
